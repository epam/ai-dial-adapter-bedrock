from typing import Callable, List, Tuple

from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)


def truncate_prompt_decorator(
    *,
    keep_message: Callable[[List[Message], int], bool],
    partitioner: Callable[[List[Message]], List[int]]
) -> ChatCompletionTransformer:
    return lambda adapter: TruncatePromptDecorator(
        adapter=adapter,
        keep_message=keep_message,
        partitioner=partitioner,
    )


class TruncatePromptDecorator(ChatCompletionDecorator):
    keep_message: Callable[[List[Message], int], bool]
    partitioner: Callable[[List[Message]], List[int]]

    async def _on_input(
        self,
        params: ModelParameters,
        messages: List[Message],
        consumer: Consumer | None = None,
    ) -> Tuple[ModelParameters, List[Message], DiscardedMessages | None]:

        async def _tokenizer(msgs: List[Message]) -> int:
            return await self.count_prompt_tokens(params, msgs)

        discarded_messages, messages = await truncate_prompt(
            messages=messages,
            tokenizer=_tokenizer,
            keep_message=self.keep_message,
            partitioner=self.partitioner,
            model_limit=None,
            user_limit=params.max_prompt_tokens,
        )

        if params.max_prompt_tokens is None:
            discarded_messages = None

        if consumer:
            consumer.set_discarded_messages(discarded_messages)

        return params, messages, discarded_messages

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        params, messages, _discarded_messages = await self._on_input(
            params, messages, consumer
        )
        await self.adapter.chat(consumer, params, messages)

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        _params, _messages, discarded_messages = await self._on_input(
            params, messages
        )
        return discarded_messages
