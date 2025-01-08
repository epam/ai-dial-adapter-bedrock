from typing import Callable, List, Tuple

from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


def tools_emulator_decorator(
    tools_emulator: Callable[[ToolsConfig], ToolsEmulator]
) -> ChatCompletionTransformer:
    return lambda adapter: ToolsEmulatorDecorator(
        tools_emulator=tools_emulator, adapter=adapter
    )


class ToolsEmulatorDecorator(ChatCompletionDecorator):
    tools_emulator: Callable[[ToolsConfig], ToolsEmulator]

    # TODO: express via a OnInput decorator
    def _on_input(
        self,
        params: ModelParameters,
        messages: List[Message],
        consumer: Consumer | None = None,
    ) -> Tuple[ModelParameters, List[Message]]:
        if params.tool_config:
            tools_emulator = self.tools_emulator(params.tool_config)

            if consumer:
                consumer.set_tools_emulator(tools_emulator)

            params = params.copy()
            params = params.add_stop_sequences(
                tools_emulator.get_stop_sequences()
            )
            params.tool_config = None

            messages = tools_emulator.parse_dial_messages(messages)

        return params, messages

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        params, messages = self._on_input(params, messages, consumer)
        await self.adapter.chat(consumer, params, messages)

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[Message]
    ) -> int:
        params, messages = self._on_input(params, messages)
        return await self.adapter.count_prompt_tokens(params, messages)

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        params, messages = self._on_input(params, messages)
        return await self.adapter.compute_discarded_messages(params, messages)
