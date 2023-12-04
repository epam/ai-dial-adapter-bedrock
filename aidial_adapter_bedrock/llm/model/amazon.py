import json
from typing import Any, Callable, Dict, Generator, List, Optional

from pydantic import BaseModel
from typing_extensions import override

import aidial_adapter_bedrock.utils.stream as stream
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation.pseudo_chat import PseudoChatConf
from aidial_adapter_bedrock.llm.chat_model import PseudoChatModel
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_AMAZON
from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class AmazonResult(BaseModel):
    tokenCount: int
    outputText: str
    completionReason: Optional[str]


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    results: List[AmazonResult]

    def content(self) -> str:
        assert (
            len(self.results) == 1
        ), "AmazonResponse should only have one result"
        return self.results[0].outputText

    def usage(self) -> TokenUsage:
        assert (
            len(self.results) == 1
        ), "AmazonResponse should only have one result"
        return TokenUsage(
            prompt_tokens=self.inputTextTokenCount,
            completion_tokens=self.results[0].tokenCount,
        )


def prepare_model_kwargs(model_params: ModelParameters) -> Dict[str, Any]:
    model_kwargs = {}

    if model_params.temperature is not None:
        model_kwargs["temperature"] = model_params.temperature

    if model_params.top_p is not None:
        model_kwargs["topP"] = model_params.top_p

    if model_params.max_tokens is not None:
        model_kwargs["maxTokenCount"] = model_params.max_tokens
    else:
        # The default for max tokens is 128, which is too small for most use cases.
        # Choosing reasonable default.
        model_kwargs["maxTokenCount"] = DEFAULT_MAX_TOKENS_AMAZON

    # NOTE: Amazon Titan (amazon.titan-tg1-large) currently only supports
    # stop sequences matching pattern "$\|+".
    # if model_params.stop is not None:
    #     model_kwargs["stopSequences"] = model_params.stop

    return model_kwargs


def prepare_input(prompt: str, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "inputText": prompt,
        "textGenerationConfig": model_kwargs,
    }


def get_generator_for_streaming(
    response: Any,
    usage: TokenUsage,
) -> Generator[str, None, None]:
    body = response["body"]
    for event in body:
        chunk = event.get("chunk")
        if chunk:
            chunk_obj = json.loads(chunk.get("bytes").decode())
            log.debug(f"chunk: {chunk_obj}")

            input_tokens = chunk_obj.get("inputTextTokenCount")
            if input_tokens is not None:
                usage.prompt_tokens = input_tokens

            output_tokens = chunk_obj.get("totalOutputTextTokenCount")
            if output_tokens is not None:
                usage.completion_tokens = output_tokens

            yield chunk_obj["outputText"]


def get_generator_for_non_streaming(
    response: Any,
    usage: TokenUsage,
) -> Generator[str, None, None]:
    body = json.loads(response["body"].read())
    log.debug(f"body: {body}")

    resp = AmazonResponse.parse_obj(body)

    token_usage = resp.usage()
    usage.completion_tokens = token_usage.completion_tokens
    usage.prompt_tokens = token_usage.prompt_tokens

    yield resp.content()


def post_process_stream(
    model_params: ModelParameters,
    content_stream: Generator[str, None, None],
    pseudo_chat_conf: PseudoChatConf,
) -> Generator[str, None, None]:
    content_stream = stream.lstrip(content_stream)

    # Titan occasionally starts its response with the role prefix
    content_stream = stream.remove_prefix(
        content_stream,
        pseudo_chat_conf.mapping["ai"] + " ",
    )

    # Titan doesn't support stop sequences, so do it manually
    if model_params.stop is not None:
        stop_sequences = (
            [model_params.stop]
            if isinstance(model_params.stop, str)
            else model_params.stop
        )
        content_stream = stream.stop_at(content_stream, stop_sequences)

    # After all the post processing, the stream may become empty.
    # To avoid this, add a space to the stream.
    content_stream = stream.ensure_not_empty(content_stream, " ")

    return content_stream


class AmazonAdapter(PseudoChatModel):
    def __init__(
        self,
        bedrock: Any,
        model_id: str,
        count_tokens: Callable[[str], int],
        pseudo_history_conf: PseudoChatConf,
    ):
        super().__init__(model_id, count_tokens, pseudo_history_conf)
        self.bedrock = bedrock

    @override
    def _validate_and_cleanup_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        messages = super()._validate_and_cleanup_messages(messages)

        # AWS Titan doesn't support empty messages,
        # so we replace it with a single space.
        for msg in messages:
            msg.content = msg.content or " "

        return messages

    async def _apredict(
        self, consumer: Consumer, model_params: ModelParameters, prompt: str
    ):
        await make_async(
            lambda args: self._call(*args), (consumer, model_params, prompt)
        )

    def _call(
        self, consumer: Consumer, model_params: ModelParameters, prompt: str
    ):
        model_kwargs = prepare_model_kwargs(model_params)

        invoke_params = {
            "modelId": self.model_id,
            "accept": "application/json",
            "contentType": "application/json",
            "body": json.dumps(prepare_input(prompt, model_kwargs)),
        }

        usage = TokenUsage()

        if not model_params.stream:
            response = self.bedrock.invoke_model(**invoke_params)
            content_stream = get_generator_for_non_streaming(response, usage)
        else:
            response = self.bedrock.invoke_model_with_response_stream(
                **invoke_params
            )
            content_stream = get_generator_for_streaming(response, usage)

        content_stream = post_process_stream(
            model_params, content_stream, self.pseudo_history_conf
        )

        for content in content_stream:
            log.debug(f"content: {repr(content)}")
            consumer.append_content(content)

        consumer.add_usage(usage)
