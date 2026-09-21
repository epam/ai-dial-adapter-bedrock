from dataclasses import dataclass

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest

from aidial_adapter_bedrock.llm.chat_model import to_dial_messages
from aidial_adapter_bedrock.llm.converse.caching import (
    get_cache_info,
)
from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)


def caching_decorator() -> ChatCompletionTransformer:
    return lambda adapter: CachingDecorator(adapter=adapter)


@dataclass
class CachingDecorator(ChatCompletionDecorator):
    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        if request.cache_breakpoint is not None:
            raise ValidationError(
                "Top-level `cache_breakpoint` is not supported because the Converse API "
                "does not support automatic caching."
            )

        tools = request.tool_config.tools if request.tool_config else []
        if info := get_cache_info(to_dial_messages(request.messages), tools):
            consumer.get_response().set_cache_breakpoint(
                cache_breakpoint_path=info.breakpoint_path,
                cache_expire_at=info.expire_at,
            )
        await self.adapter.chat(consumer, request)
