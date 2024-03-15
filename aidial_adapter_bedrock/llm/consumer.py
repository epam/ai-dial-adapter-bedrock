from abc import ABC, abstractmethod
from typing import List, Optional, assert_never

from aidial_sdk.chat_completion import Choice, FinishReason
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
    def close_content(self, finish_reason: FinishReason = FinishReason.STOP):
        pass

    @abstractmethod
    def add_attachment(self, attachment: Attachment):
        pass

    @abstractmethod
    def add_usage(self, usage: TokenUsage):
        pass

    @abstractmethod
    def set_discarded_messages(self, discarded_messages: List[int]):
        pass


class ChoiceConsumer(Consumer):
    usage: TokenUsage
    choice: Choice
    discarded_messages: Optional[List[int]]
    tools_emulator: ToolsEmulator

    def __init__(self, tools_emulator: ToolsEmulator, choice: Choice):
        self.choice = choice
        self.usage = TokenUsage()
        self.discarded_messages = None
        self.tools_emulator = tools_emulator

    def _process_content(
        self, content: str | None, finish_reason: FinishReason | None = None
    ):
        res = self.tools_emulator.recognize_call(content)

        if res is None:
            self.choice._last_finish_reason = finish_reason
            return

        if isinstance(res, str):
            self.choice.append_content(res)
            return

        if isinstance(res, AIToolCallMessage):
            for call in res.calls:
                self.choice.create_function_tool_call(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
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

    def set_discarded_messages(self, discarded_messages: List[int]):
        self.discarded_messages = discarded_messages

    def finish(self, finish_reason: FinishReason):
        self.choice.close(finish_reason)
