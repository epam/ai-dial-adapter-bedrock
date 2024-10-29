from enum import Enum
from typing import Any, Literal, TypedDict


class ConverseRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConverseToolConfig(TypedDict):
    name: str
    description: str
    inputSchema: dict


class ConverseTools(TypedDict):
    tools: list[ConverseToolConfig]


class ConverseToolUseConfig(TypedDict):
    toolUseId: str
    name: str
    #  {...}|[...]|123|123.4|'string'|True|None
    input: Any


class ConverseToolUse(TypedDict):
    toolUse: ConverseToolUseConfig


class ConverseToolResultConfig(TypedDict):
    toolUseId: str
    content: list[dict]
    status: str


class ConverseToolResult(TypedDict):
    toolResult: ConverseToolResultConfig


class ConverseStopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    GUARDRAIL_INTERVENED = "guardrail_intervened"
    CONTENT_FILTERED = "content_filtered"


class ConverseImageSource(TypedDict):
    bytes: bytes


class ConverseImagePart(TypedDict):
    format: Literal["png", "jpeg", "gif", "webp"]
    source: ConverseImageSource


class ConverseContentPart(TypedDict, total=False):
    text: str | None
    image: ConverseImagePart | None
    toolUse: ConverseToolUseConfig | None
    toolResult: ConverseToolResultConfig | None


class ConverseMessage(TypedDict):
    role: ConverseRole
    content: list[ConverseContentPart]
