from collections.abc import Callable
from dataclasses import dataclass

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


@dataclass
class ChatCompletionDecorator(ChatCompletionAdapter):
    adapter: ChatCompletionAdapter

    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        await self.adapter.chat(consumer, request)

    async def configuration(self) -> type[BaseModel]:
        return await self.adapter.configuration()

    async def count_prompt_tokens(self, request: AdapterRequest) -> int:
        return await self.adapter.count_prompt_tokens(request)

    async def count_completion_tokens(self, string: str) -> int:
        return await self.adapter.count_completion_tokens(string)

    async def compute_discarded_messages(
        self, request: AdapterRequest
    ) -> DiscardedMessages | None:
        return await self.adapter.compute_discarded_messages(request)


ChatCompletionTransformer = Callable[
    [ChatCompletionAdapter], ChatCompletionAdapter
]


def compose_decorators(
    *decorators: ChatCompletionTransformer,
) -> ChatCompletionTransformer:
    def compose(adapter: ChatCompletionAdapter) -> ChatCompletionAdapter:
        for decorator in reversed(decorators):
            adapter = decorator(adapter)
        return adapter

    return compose
