from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypedDict, Union

from aidial_adapter_bedrock.utils.list_projection import ListProjection


class ConverseRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConverseTextPart(TypedDict):
    text: str


class ConverseJsonPart(TypedDict):
    json: dict


class ConverseImageSource(TypedDict):
    bytes: bytes


class ConverseImagePartConfig(TypedDict):
    format: Literal["png", "jpeg", "gif", "webp"]
    source: ConverseImageSource


class ConverseImagePart(TypedDict):
    image: ConverseImagePartConfig


class ConverseToolUseConfig(TypedDict):
    toolUseId: str
    name: str
    #  {...}|[...]|123|123.4|'string'|True|None
    input: Any


class ConverseToolUsePart(TypedDict):
    toolUse: ConverseToolUseConfig


class ConverseToolResultConfig(TypedDict):
    toolUseId: str
    content: list[ConverseTextPart | ConverseJsonPart]
    status: str


class ConverseToolResultPart(TypedDict):
    toolResult: ConverseToolResultConfig


ConverseContentPart = Union[
    ConverseTextPart,
    ConverseJsonPart,
    ConverseImagePart,
    ConverseToolUsePart,
    ConverseToolResultPart,
]


class ConverseToolConfig(TypedDict):
    name: str
    description: str
    inputSchema: dict


class ConverseToolSpec(TypedDict):
    toolSpec: ConverseToolConfig


class ConverseTools(TypedDict):
    tools: list[ConverseToolSpec]
    toolChoice: dict


class ConverseToolUse(TypedDict):
    toolUse: ConverseToolUseConfig


class ConverseStopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    GUARDRAIL_INTERVENED = "guardrail_intervened"
    CONTENT_FILTERED = "content_filtered"


class ConverseMessage(TypedDict):
    role: ConverseRole
    content: list[ConverseContentPart]


class InferenceConfig(TypedDict, total=False):
    temperature: float | None
    topP: float | None
    maxTokens: int | None
    stopSequences: list[str] | None


class ConverseRequest(TypedDict, total=False):
    messages: list[ConverseMessage]
    system: list[ConverseTextPart] | None
    inferenceConfig: InferenceConfig | None
    toolConfig: ConverseTools | None


@dataclass
class ConverseRequestWrapper:
    messages: ListProjection[ConverseMessage]
    system: list[ConverseTextPart] | None = None
    inferenceConfig: InferenceConfig | None = None
    toolConfig: ConverseTools | None = None

    def to_request(self) -> ConverseRequest:
        return ConverseRequest(
            system=self.system,
            messages=self.messages.raw_list,
            inferenceConfig=self.inferenceConfig,
            toolConfig=self.toolConfig,
        )


ConverseDeployment = str
