from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.tools.base import ToolConfig


class ToolsEmulator(ABC, BaseModel):
    tool_config: ToolConfig

    @abstractmethod
    def transform_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        pass
