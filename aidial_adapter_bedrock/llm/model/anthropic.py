import json
from typing import Any, Dict, Generator, List, Optional

from anthropic.tokenizer import count_tokens

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation import claude_chat
from aidial_adapter_bedrock.llm.chat_emulation.claude_chat import (
    ClaudeChatHistory,
)
from aidial_adapter_bedrock.llm.chat_model import ChatModel, ChatPrompt
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def compute_usage(prompt: str, completion: str) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=count_tokens(prompt),
        completion_tokens=count_tokens(completion),
    )


# NOTE: See https://docs.anthropic.com/claude/reference/complete_post
def prepare_model_kwargs(params: ModelParameters) -> Dict[str, Any]:
    ret = {}

    if params.max_tokens is not None:
        ret["max_tokens_to_sample"] = params.max_tokens
    else:
        # The max tokens parameter is required for Anthropic models.
        # Choosing reasonable default.
        ret["max_tokens_to_sample"] = DEFAULT_MAX_TOKENS_ANTHROPIC

    if params.stop:
        ret["stop_sequences"] = params.stop

    if params.temperature is not None:
        ret["temperature"] = params.temperature

    if params.top_p is not None:
        ret["top_p"] = params.top_p

    return ret


def prepare_input(prompt: str, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **model_kwargs}


def get_generator_for_streaming(response: Any) -> Generator[str, None, None]:
    body = response["body"]
    for event in body:
        chunk = event.get("chunk")
        if chunk:
            chunk_obj = json.loads(chunk.get("bytes").decode())
            log.debug(f"chunk: {chunk_obj}")

            yield chunk_obj["completion"]


def get_generator_for_non_streaming(
    response: Any,
) -> Generator[str, None, None]:
    body = json.loads(response["body"].read())
    log.debug(f"body: {body}")
    yield body["completion"]


class AnthropicAdapter(ChatModel):
    def __init__(
        self,
        bedrock: Any,
        model_id: str,
    ):
        super().__init__(model_id)
        self.bedrock = bedrock

    def _prepare_prompt(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> ChatPrompt:
        history = ClaudeChatHistory.create(messages)
        if max_prompt_tokens is None:
            return ChatPrompt(
                text=history.format(), stop_sequences=claude_chat.STOP_SEQUENCES
            )

        history, discarded_messages_count = history.trim(
            count_tokens, max_prompt_tokens
        )

        return ChatPrompt(
            text=history.format(),
            stop_sequences=claude_chat.STOP_SEQUENCES,
            discarded_messages=discarded_messages_count,
        )

    async def _apredict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        return await make_async(
            lambda args: self._predict(*args), (consumer, params, prompt)
        )

    def _predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        model_kwargs = prepare_model_kwargs(params)

        invoke_params = {
            "modelId": self.model,
            "accept": "application/json",
            "contentType": "application/json",
            "body": json.dumps(prepare_input(prompt, model_kwargs)),
        }

        if not params.stream:
            response = self.bedrock.invoke_model(**invoke_params)
            content_stream = get_generator_for_non_streaming(response)

        else:
            response = self.bedrock.invoke_model_with_response_stream(
                **invoke_params
            )
            content_stream = get_generator_for_streaming(response)

        completion = ""

        for content in content_stream:
            completion += content
            consumer.append_content(content)

        consumer.add_usage(compute_usage(prompt, completion))
