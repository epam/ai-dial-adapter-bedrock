from abc import ABC, abstractmethod
from typing import Optional, assert_never

from aidial_sdk.chat_completion import (
    Choice,
    FinishReason,
    FunctionCall,
    ToolCall,
)
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
)
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


class Attachment(BaseModel):
    type: str | None = None
    title: str | None = None
    data: str | None = None
    url: str | None = None
    reference_url: str | None = None
    reference_type: str | None = None


class Consumer(ABC):
    @abstractmethod
    def append_content(self, content: str):
        pass

    @abstractmethod
    def close_content(self, finish_reason: FinishReason | None = None):
        pass

    @abstractmethod
    def add_attachment(self, attachment: Attachment):
        pass

    @abstractmethod
    def add_usage(self, usage: TokenUsage):
        pass

    @abstractmethod
    def set_discarded_messages(
        self, discarded_messages: Optional[DiscardedMessages]
    ):
        pass

    @abstractmethod
    def create_function_tool_call(self, tool_call: ToolCall):
        pass

    @abstractmethod
    def create_function_call(self, function_call: FunctionCall):
        pass

    @property
    @abstractmethod
    def has_function_call(self) -> bool:
        pass


class ChoiceConsumer(Consumer):
    usage: TokenUsage
    choice: Choice
    discarded_messages: Optional[DiscardedMessages]
    tools_emulator: Optional[ToolsEmulator]

    def __init__(self, choice: Choice):
        self.choice = choice
        self.usage = TokenUsage()
        self.discarded_messages = None
        self.tools_emulator = None

    def set_tools_emulator(self, tools_emulator: ToolsEmulator):
        self.tools_emulator = tools_emulator

    def _process_content(
        self, content: str | None, finish_reason: FinishReason | None = None
    ):
        if self.tools_emulator is not None:
            res = self.tools_emulator.recognize_call(content)
        else:
            res = content

        if res is None:
            # Choice.close(finish_reason: Optional[FinishReason]) can be called only once
            # Currently, there's no other way to explicitly set the finish reason
            self.choice._last_finish_reason = finish_reason
            return

        if isinstance(res, str):
            self.choice.append_content(res)
            return

        if isinstance(res, AIToolCallMessage):
            for call in res.calls:
                self.create_function_tool_call(call)
            return

        if isinstance(res, AIFunctionCallMessage):
            call = res.call
            self.choice.create_function_call(
                name=call.name, arguments=call.arguments
            )
            return

        assert_never(res)

    def close_content(self, finish_reason: FinishReason | None = None):
        self._process_content(None, finish_reason)

    def append_content(self, content: str):
        self._process_content(content)

    def add_attachment(self, attachment: Attachment):
        self.choice.add_attachment(**attachment.dict())

    def add_usage(self, usage: TokenUsage):
        self.usage.accumulate(usage)

    def set_discarded_messages(
        self, discarded_messages: Optional[DiscardedMessages]
    ):
        self.discarded_messages = discarded_messages

    def create_function_tool_call(self, tool_call: ToolCall):
        self.choice.create_function_tool_call(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )

    def create_function_call(self, function_call: FunctionCall):
        self.choice.create_function_call(
            name=function_call.name, arguments=function_call.arguments
        )

    @property
    def has_function_call(self) -> bool:
        return self.choice.has_function_call
