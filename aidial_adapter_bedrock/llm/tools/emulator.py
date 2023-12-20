from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
    BaseMessage,
    ToolMessage,
)
from aidial_adapter_bedrock.llm.tools.tool_config import ToolConfig


class ToolsEmulator(ABC, BaseModel):
    tool_config: Optional[ToolConfig]

    @abstractmethod
    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[str]]:
        pass

    @abstractmethod
    def convert_to_base_messages(
        self, messages: List[BaseMessage | ToolMessage]
    ) -> List[BaseMessage]:
        pass

    @abstractmethod
    def recognize_call(
        self, content: str | None
    ) -> str | AIToolCallMessage | AIFunctionCallMessage | None:
        pass
