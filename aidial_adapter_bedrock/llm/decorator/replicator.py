import asyncio
from dataclasses import replace

from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import AdapterRequest

from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)


def replicator_decorator() -> ChatCompletionTransformer:
    return lambda adapter: ReplicatorDecorator(adapter=adapter)


class ReplicatorDecorator(ChatCompletionDecorator):
    async def chat(self, consumer: Consumer, request: AdapterRequest) -> None:
        single = replace(request, n=1)

        async def _chat(root_consumer: Consumer):
            async with root_consumer.fork() as consumer:
                await self.adapter.chat(consumer, single)

        await asyncio.gather(*(_chat(consumer) for _ in range(request.n)))
