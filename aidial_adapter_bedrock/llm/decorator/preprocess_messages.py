from collections.abc import Callable
from dataclasses import dataclass, replace

from aidial_adapter_anthropic.dial._message import AdapterMessage
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest

from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from aidial_adapter_bedrock.utils.list_projection import ListProjection

OnMessages = Callable[
    [ListProjection[AdapterMessage]], ListProjection[AdapterMessage]
]


def preprocess_messages_decorator(
    on_messages: OnMessages,
) -> ChatCompletionTransformer:
    return lambda adapter: PreprocessMessagesDecorator(
        on_messages=on_messages, adapter=adapter
    )


@dataclass
class PreprocessMessagesDecorator(ChatCompletionDecorator):
    on_messages: OnMessages

    def _preprocess(self, request: AdapterRequest) -> AdapterRequest:
        return replace(request, messages=self.on_messages(request.messages))

    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        await self.adapter.chat(consumer, self._preprocess(request))

    async def count_prompt_tokens(self, request: AdapterRequest) -> int:
        return await self.adapter.count_prompt_tokens(self._preprocess(request))

    async def compute_discarded_messages(
        self, request: AdapterRequest
    ) -> DiscardedMessages | None:
        return await self.adapter.compute_discarded_messages(
            self._preprocess(request)
        )
