from abc import ABC, abstractmethod
from typing import List, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.tools.base import ToolConfig


class ToolsEmulator(ABC, BaseModel):
    tool_config: ToolConfig

    @abstractmethod
    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[str]]:
        pass
