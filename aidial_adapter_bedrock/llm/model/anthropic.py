import json
from typing import Any, Dict, Generator, List, Optional

from anthropic.tokenizer import count_tokens

from aidial_adapter_bedrock.llm.chat_emulation import claude_chat
from aidial_adapter_bedrock.llm.chat_emulation.claude_chat import (
    ClaudeChatHistory,
)
from aidial_adapter_bedrock.llm.chat_model import ChatModel, ChatPrompt
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.universal_api.request import ModelParameters
from aidial_adapter_bedrock.universal_api.token_usage import TokenUsage
from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def compute_usage(prompt: str, completion: str) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=count_tokens(prompt),
        completion_tokens=count_tokens(completion),
    )


# NOTE: See https://docs.anthropic.com/claude/reference/complete_post
def prepare_model_kwargs(model_params: ModelParameters) -> Dict[str, Any]:
    model_kwargs = {}

    if model_params.max_tokens is not None:
        model_kwargs["max_tokens_to_sample"] = model_params.max_tokens
    else:
        # The max tokens parameter is required for Anthropic models.
        # Choosing reasonable default.
        model_kwargs["max_tokens_to_sample"] = DEFAULT_MAX_TOKENS_ANTHROPIC

    if model_params.stop is not None:
        model_kwargs["stop_sequences"] = (
            [model_params.stop]
            if isinstance(model_params.stop, str)
            else model_params.stop
        )

    if model_params.temperature is not None:
        model_kwargs["temperature"] = model_params.temperature

    if model_params.top_p is not None:
        model_kwargs["top_p"] = model_params.top_p

    return model_kwargs


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
        self, consumer: Consumer, model_params: ModelParameters, prompt: str
    ):
        return await make_async(
            lambda args: self._predict(*args), (consumer, model_params, prompt)
        )

    def _predict(
        self, consumer: Consumer, model_params: ModelParameters, prompt: str
    ):
        model_kwargs = prepare_model_kwargs(model_params)

        invoke_params = {
            "modelId": self.model_id,
            "accept": "application/json",
            "contentType": "application/json",
            "body": json.dumps(prepare_input(prompt, model_kwargs)),
        }

        if not model_params.stream:
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
