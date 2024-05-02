from abc import ABC, abstractmethod
from typing import List, Optional

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
    BaseMessage,
    ToolMessage,
    parse_dial_message,
)
from aidial_adapter_bedrock.llm.tools.tool_config import ToolConfig


class ToolsEmulator(ABC, BaseModel):
    tool_config: Optional[ToolConfig]

    @abstractmethod
    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        Adds description of the given tools into the chat messages.
        Typically the tools description is added as a first system message,
        which describes the tools protocol.
        """

    @abstractmethod
    def get_stop_sequences(self) -> List[str]:
        """
        Return stop sequence for the model response.
        Typically it's a marker for an end of a function response:
        after the model has generated function call, we are not
        interested in anything else which may follow.
        """

    @abstractmethod
    def convert_to_base_messages(
        self, messages: List[BaseMessage | ToolMessage]
    ) -> List[BaseMessage]:
        """
        Converting a list of non-function messages (BaseMessage) and
        function messages (ToolMessage) into a list of non-function messages.
        It involves conversion of a function message to a non-function message
        following certain protocol described beforehand in a system message.
        """

    @abstractmethod
    def recognize_call(
        self, content: str | None
    ) -> str | AIToolCallMessage | AIFunctionCallMessage | None:
        """
        Recognizing function/tool call from a model response.
        """

    def parse_dial_messages(self, messages: List[Message]) -> List[BaseMessage]:
        parsed_messages = list(map(parse_dial_message, messages))
        base_messages = self.convert_to_base_messages(parsed_messages)
        return self.add_tool_declarations(base_messages)
