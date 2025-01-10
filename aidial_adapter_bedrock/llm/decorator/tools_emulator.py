from typing import Callable, List, Tuple, assert_never

from aidial_sdk.chat_completion import FinishReason, Message

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.consumer import Consumer, ConsumerDecorator
from aidial_adapter_bedrock.llm.decorator.base import (
    ChatCompletionDecorator,
    ChatCompletionTransformer,
)
from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
)
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


def tools_emulator_decorator(
    tools_emulator_factor: Callable[[ToolsConfig], ToolsEmulator]
) -> ChatCompletionTransformer:
    return lambda adapter: ToolsEmulatorDecorator(
        tools_emulator_factory=tools_emulator_factor, adapter=adapter
    )


class ToolsConsumer(ConsumerDecorator):
    tools_emulator: ToolsEmulator

    def __init__(self, consumer: Consumer, tools_emulator: ToolsEmulator):
        super().__init__(consumer)
        self.tools_emulator = tools_emulator

    def _process_content(
        self, content: str | None, finish_reason: FinishReason | None
    ):
        match res := self.tools_emulator.recognize_call(content):
            case None:
                self.consumer.close_content(finish_reason)

            case str():
                self.consumer.append_content(res)

            case AIToolCallMessage(calls=calls):
                for call in calls:
                    self.consumer.create_function_tool_call(call)

            case AIFunctionCallMessage(call=call):
                self.consumer.create_function_call(call)

            case _:
                assert_never(res)

    def close_content(self, finish_reason: FinishReason | None = None):
        self._process_content(None, finish_reason)

    def append_content(self, content: str):
        self._process_content(content, None)


class ToolsEmulatorDecorator(ChatCompletionDecorator):
    tools_emulator_factory: Callable[[ToolsConfig], ToolsEmulator]

    def _on_input(
        self, params: ModelParameters, messages: List[Message]
    ) -> Tuple[ModelParameters, List[Message]]:
        if params.tool_config:
            tools_emulator = self.tools_emulator_factory(params.tool_config)

            params = params.copy()
            params.stop = params.stop + tools_emulator.get_stop_sequences()
            params.tool_config = None

            messages = tools_emulator.parse_dial_messages(messages)

        return params, messages

    def _on_consumer(
        self, params: ModelParameters, consumer: Consumer
    ) -> Consumer:
        if params.tool_config:
            tools_emulator = self.tools_emulator_factory(params.tool_config)
            return ToolsConsumer(consumer, tools_emulator)
        return consumer

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:
        consumer = self._on_consumer(params, consumer)
        params, messages = self._on_input(params, messages)
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
