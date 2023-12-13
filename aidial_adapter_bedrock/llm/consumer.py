from abc import ABC, abstractmethod
from typing import Optional

from aidial_sdk.chat_completion import Choice, FunctionCall, ToolCall
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.tools.base import ToolsMode
from aidial_adapter_bedrock.llm.tools.claude import parse_function_call
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

    def append_content(self, content: str):
        tool_config = self.tools_emulator.tool_config
        if tool_config is None:
            self.choice.append_content(content)
            return

        # FIXME: support streaming mode
        call: Optional[FunctionCall] = None
        try:
            call = parse_function_call(content)
        except Exception:
            pass

        if call is None:
            self.choice.append_content(content)
            return

        if tool_config.mode == ToolsMode.FUNCTIONS:
            self.choice.add_function_call(call)
        else:
            self.choice.add_tool_calls(
                [ToolCall(id=call.name, type="function", function=call)]
            )

    def add_attachment(self, attachment: Attachment):
        self.choice.add_attachment(**attachment.dict())

    def add_usage(self, usage: TokenUsage):
        self.usage.accumulate(usage)

    def set_discarded_messages(self, discarded_messages: int):
        self.discarded_messages = discarded_messages
