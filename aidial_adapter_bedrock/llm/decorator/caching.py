from dataclasses import dataclass

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.llm.converse.caching import (
    get_response_headers_for_caching,
)
from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)


def caching_decorator() -> ChatCompletionTransformer:
    return lambda adapter: CachingDecorator(adapter=adapter)


@dataclass
class CachingDecorator(ChatCompletionDecorator):
    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: list[Message],
    ) -> None:
        if params.cache_breakpoint is not None:
            raise ValidationError(
                "Top-level `cache_breakpoint` is not supported because the Converse API "
                "does not support automatic caching."
            )

        tools = params.tool_config.tools if params.tool_config else []
        if headers := get_response_headers_for_caching(messages, tools):
            await consumer.set_response_headers(headers)
        await self.adapter.chat(consumer, params, messages)
