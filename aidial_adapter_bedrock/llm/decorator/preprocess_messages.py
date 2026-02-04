from dataclasses import dataclass
from typing import Callable, List

from aidial_adapter_anthropic.dial.consumer import Consumer
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from aidial_adapter_bedrock.utils.list_projection import ListProjection


def preprocess_messages_decorator(
    on_messages: Callable[[List[Message]], ListProjection[Message]],
) -> ChatCompletionTransformer:
    return lambda adapter: PreprocessMessagesDecorator(
        on_messages=on_messages, adapter=adapter
    )


@dataclass
class PreprocessMessagesDecorator(ChatCompletionDecorator):
    on_messages: Callable[[List[Message]], ListProjection[Message]]

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        new_messages = self.on_messages(messages)
        await self.adapter.chat(consumer, params, new_messages.raw_list)
        if (
            discarded_messages := consumer.get_discarded_messages()
        ) is not None:
            discarded_messages = list(
                new_messages.to_original_indices(discarded_messages)
            )
            consumer.set_discarded_messages(discarded_messages)

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        new_messages = self.on_messages(messages)
        return await self.adapter.count_prompt_tokens(
            params, new_messages.raw_list
        )

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        new_messages = self.on_messages(messages)
        discarded_messages = await self.adapter.compute_discarded_messages(
            params, new_messages.raw_list
        )

        if discarded_messages is not None:
            discarded_messages = list(
                new_messages.to_original_indices(discarded_messages)
            )

        return discarded_messages
