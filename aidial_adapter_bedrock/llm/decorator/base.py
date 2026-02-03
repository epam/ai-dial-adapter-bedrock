from dataclasses import dataclass
from typing import Callable, List

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


@dataclass
class ChatCompletionDecorator(ChatCompletionAdapter):
    adapter: ChatCompletionAdapter

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        await self.adapter.chat(consumer, params, messages)

    async def configuration(self) -> type[BaseModel]:
        return await self.adapter.configuration()

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        return await self.adapter.count_prompt_tokens(params, messages)

    async def count_completion_tokens(self, string: str) -> int:
        return await self.adapter.count_completion_tokens(string)

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        return await self.adapter.compute_discarded_messages(params, messages)


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
