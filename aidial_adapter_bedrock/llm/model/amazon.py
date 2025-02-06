from typing import Any, AsyncIterator, Dict, List, Optional

from aidial_sdk.chat_completion import FinishReason, Message
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulator import (
    default_emulator,
    post_process_completion_stream,
)
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    TextCompletionAdapter,
    default_preprocess_messages,
    keep_last_and_system_messages,
    trivial_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.pseudo_chat import pseudo_chat_adapter
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.decorator.tools_emulator import (
    tools_emulator_decorator,
)
from aidial_adapter_bedrock.llm.decorator.truncate_prompt import (
    truncate_prompt_decorator,
)
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_AMAZON
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)
from aidial_adapter_bedrock.utils.list_projection import ListProjection


class AmazonResult(BaseModel):
    tokenCount: int
    outputText: str
    completionReason: Optional[str]


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    results: List[AmazonResult]

    def content(self) -> str:
        return self.result.outputText

    def usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.inputTextTokenCount,
            completion_tokens=self.result.tokenCount,
        )

    def finish_reason(self) -> FinishReason | None:
        if self.result.completionReason:
            return _to_dial_finish_reason(self.result.completionReason)
        return None

    @property
    def result(self) -> AmazonResult:
        assert (
            len(self.results) == 1
        ), "AmazonResponse should only have one result"
        return self.results[0]


def _to_dial_finish_reason(reason: str) -> FinishReason | None:
    match reason:
        case "FINISHED" | "STOP_CRITERIA_MET":
            return FinishReason.STOP
        case "LENGTH":
            return FinishReason.LENGTH
        case "CONTENT_FILTERED":
            return FinishReason.CONTENT_FILTER
        case _:
            return None


def convert_params(params: ModelParameters) -> Dict[str, Any]:
    ret = {}

    if params.temperature is not None:
        ret["temperature"] = params.temperature

    if params.top_p is not None:
        ret["topP"] = params.top_p

    if params.max_tokens is not None:
        ret["maxTokenCount"] = params.max_tokens
    else:
        # The default for max tokens is 128, which is too small for most use cases.
        # Choosing reasonable default.
        ret["maxTokenCount"] = DEFAULT_MAX_TOKENS_AMAZON

    # NOTE: Amazon Titan (amazon.titan-tg1-large) currently only supports
    # stop sequences matching pattern "$\|+".
    # if params.stop is not None:
    #     ret["stopSequences"] = params.stop

    return ret


def create_request(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"inputText": prompt, "textGenerationConfig": params}


FinishReasons = Dict[int, FinishReason]


async def chunks_to_stream(
    chunks: AsyncIterator[dict],
    usage: TokenUsage,
    finish_reasons: FinishReasons,
) -> AsyncIterator[str]:
    async for chunk in chunks:
        input_tokens = chunk.get("inputTextTokenCount")
        if input_tokens is not None:
            usage.prompt_tokens = input_tokens

        output_tokens = chunk.get("totalOutputTextTokenCount")
        if output_tokens is not None:
            usage.completion_tokens = output_tokens

        if completionReason := chunk.get("completionReason"):
            finish_reason = _to_dial_finish_reason(completionReason)
            index = chunk.get("index") or 0
            if finish_reason:
                finish_reasons[index] = finish_reason

        yield chunk["outputText"]


async def response_to_stream(
    response: dict,
    usage: TokenUsage,
    finish_reasons: FinishReasons,
) -> AsyncIterator[str]:
    resp = AmazonResponse.parse_obj(response)

    if finish_reason := resp.finish_reason():
        finish_reasons[0] = finish_reason

    token_usage = resp.usage()
    usage.completion_tokens = token_usage.completion_tokens
    usage.prompt_tokens = token_usage.prompt_tokens

    yield resp.content()


def _preprocess_amazon_messages(
    messages: List[Message],
) -> ListProjection[Message]:
    ret = default_preprocess_messages(messages)

    # AWS Titan doesn't support empty messages,
    # so we replace it with a single space.
    for msg in ret.raw_list:
        msg.content = msg.content or " "

    return ret


def create_adapter(client: Bedrock, model: str) -> ChatCompletionAdapter:
    return compose_decorators(
        preprocess_messages_decorator(_preprocess_amazon_messages),
        truncate_prompt_decorator(
            keep_message=keep_last_and_system_messages,
            partitioner=trivial_partitioner,
        ),
        replicator_decorator(),
        tools_emulator_decorator(default_tools_emulator),
    )(
        # TODO: To use conversational mode on Titan, you can use the format of User: {{}} \n Bot: when prompting the model.
        # See the note at the end of: https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-a-prompt.html
        pseudo_chat_adapter(default_emulator)(
            AmazonAdapter(client=client, model=model)
        )
    )


class AmazonAdapter(TextCompletionAdapter):
    model: str
    client: Bedrock

    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt, convert_params(params))

        usage = TokenUsage()
        finish_reasons: FinishReasons = {}

        if params.stream:
            chunks = self.client.ainvoke_streaming(self.model, args)
            stream = chunks_to_stream(chunks, usage, finish_reasons)
        else:
            response, _headers = await self.client.ainvoke_non_streaming(
                self.model, args
            )
            stream = response_to_stream(response, usage, finish_reasons)

        stream = post_process_completion_stream(
            params, default_emulator, stream
        )

        async for content in stream:
            consumer.append_content(content)

        finish_reason = next((r for r in finish_reasons.values()), None)
        consumer.close_content(finish_reason)

        consumer.add_usage(usage)

    async def count_completion_tokens(self, string: str) -> int:
        return default_tokenize_string(string)

    async def count_prompt_tokens(
        self, params: ModelParameters, prompt: str
    ) -> int:
        return default_tokenize_string(prompt)
