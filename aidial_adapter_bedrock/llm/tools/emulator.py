from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
    BaseMessage,
    ToolMessage,
)
from aidial_adapter_bedrock.llm.tools.base import ToolConfig
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


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


class DefaultToolsEmulator(ToolsEmulator):
    def add_tool_declarations(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[BaseMessage], List[str]]:
        if self.tool_config is not None:
            log.warning(
                "The model doesn't support tools/functions, however they were specified in the request. Continuing without tools/functions."
            )
        return messages, []

    def convert_to_base_messages(
        self,
        messages: List[BaseMessage | ToolMessage],
    ) -> List[BaseMessage]:
        ret: List[BaseMessage] = [
            msg for msg in messages if not isinstance(msg, ToolMessage)
        ]
        if len(ret) != len(messages):
            log.warning(
                "The model doesn't support tools/functions, however messages related to tools/functions were provided in the request. Ignoring such messages."
            )
        return ret

    def recognize_call(
        self, content: str | None
    ) -> str | AIToolCallMessage | AIFunctionCallMessage | None:
        return content


def default_tools_emulator(tool_config: Optional[ToolConfig]) -> ToolsEmulator:
    return DefaultToolsEmulator(tool_config=tool_config)
