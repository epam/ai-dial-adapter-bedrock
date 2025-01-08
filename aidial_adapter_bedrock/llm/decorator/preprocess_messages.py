from typing import Callable, List

from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


def preprocess_messages_decorator(
    on_messages: Callable[[List[Message]], List[Message]]
) -> ChatCompletionTransformer:
    return lambda adapter: PreprocessMessagesDecorator(
        on_messages=on_messages, adapter=adapter
    )


class PreprocessMessagesDecorator(ChatCompletionDecorator):
    on_messages: Callable[[List[Message]], List[Message]]

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        await self.adapter.chat(consumer, params, self.on_messages(messages))

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        return await self.adapter.count_prompt_tokens(
            params, self.on_messages(messages)
        )

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        return await self.adapter.compute_discarded_messages(
            params, self.on_messages(messages)
        )
