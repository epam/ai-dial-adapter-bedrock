from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.tools_emulation.base import ToolConfig
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class ToolsEmulator(ABC, BaseModel):
    tool_config: Optional[ToolConfig]

    @abstractmethod
    def transform_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        pass


class NoOpToolsEmulator(ToolsEmulator):
    def transform_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        if self.tool_config is not None:
            log.warning(
                "The model doesn't support tools/functions, however they were specified in the request. Continuing without tools/functions."
            )

        return messages
