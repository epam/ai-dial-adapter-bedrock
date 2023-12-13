from abc import ABC, abstractmethod
from typing import Optional, assert_never

from aidial_sdk.chat_completion import Choice
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
)
from aidial_adapter_bedrock.llm.tools.emulator import ToolsEmulator


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
    def close_content(self):
        pass

    @abstractmethod
    def add_attachment(self, attachment: Attachment):
        pass

    @abstractmethod
    def add_usage(self, usage: TokenUsage):
        pass

    @abstractmethod
    def set_discarded_messages(self, discarded_messages: int):
        pass


class ChoiceConsumer(Consumer):
    usage: TokenUsage
    choice: Choice
    discarded_messages: Optional[int]
    tools_emulator: ToolsEmulator

    def __init__(self, tools_emulator: ToolsEmulator, choice: Choice):
        self.choice = choice
        self.usage = TokenUsage()
        self.discarded_messages = None
        self.tools_emulator = tools_emulator

    def _process_content(self, content: str | None):
        res = self.tools_emulator.recognize_call(content)

        if res is None:
            return

        if isinstance(res, str):
            self.choice.append_content(res)
            return

        if isinstance(res, AIToolCallMessage):
            self.choice.add_tool_calls(res.calls)
            return

        if isinstance(res, AIFunctionCallMessage):
            self.choice.add_function_call(res.call)
            return

        assert_never(res)

    def close_content(self):
        self._process_content(None)

    def append_content(self, content: str):
        self._process_content(content)

    def add_attachment(self, attachment: Attachment):
        self.choice.add_attachment(**attachment.dict())

    def add_usage(self, usage: TokenUsage):
        self.usage.accumulate(usage)

    def set_discarded_messages(self, discarded_messages: int):
        self.discarded_messages = discarded_messages
