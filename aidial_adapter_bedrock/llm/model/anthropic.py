from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic.tokenizer import count_tokens

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.chat_emulation import claude_chat
from aidial_adapter_bedrock.llm.chat_emulation.claude_chat import (
    ClaudeChatHistory,
)
from aidial_adapter_bedrock.llm.chat_model import ChatModel, ChatPrompt
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC


def compute_usage(prompt: str, completion: str) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=count_tokens(prompt),
        completion_tokens=count_tokens(completion),
    )


# NOTE: See https://docs.anthropic.com/claude/reference/complete_post
def convert_params(params: ModelParameters) -> Dict[str, Any]:
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


def create_request(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **params}


async def chunks_to_stream(
    chunks: AsyncIterator[dict],
) -> AsyncIterator[str]:
    async for chunk in chunks:
        yield chunk["completion"]


async def response_to_stream(response: dict) -> AsyncIterator[str]:
    yield response["completion"]


class AnthropicAdapter(ChatModel):
    client: Bedrock

    def __init__(self, client: Bedrock, model: str):
        super().__init__(model)
        self.client = client

    def _prepare_prompt(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> ChatPrompt:
        is_claude_2_1 = (
            self.model == BedrockDeployment.ANTHROPIC_CLAUDE_V2_1_200K
        )
        history = ClaudeChatHistory.create(
            messages, system_message_is_supported=is_claude_2_1
        )

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
        args = create_request(prompt, convert_params(params))

        if params.stream:
            chunks = self.client.ainvoke_streaming(self.model, args)
            stream = chunks_to_stream(chunks)
        else:
            response = await self.client.ainvoke_non_streaming(self.model, args)
            stream = response_to_stream(response)

        stream = stream_utils.lstrip(stream)

        completion = ""
        async for content in stream:
            completion += content
            consumer.append_content(content)

        consumer.add_usage(compute_usage(prompt, completion))
