"""
Types for Converse API:
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Required, TypedDict, Union

from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.list_projection import ListProjection


class ConverseRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConverseTextPart(TypedDict):
    text: str


class ConverseJsonPart(TypedDict):
    json: dict


class ConverseSource(TypedDict):
    bytes: bytes


class ConverseImagePartConfig(TypedDict):
    format: Literal["png", "jpeg", "gif", "webp"] | str
    source: ConverseSource


class ConverseImagePart(TypedDict):
    image: ConverseImagePartConfig


class ConverseDocumentPartConfig(TypedDict):
    format: (
        Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
        | str
    )
    name: str
    source: ConverseSource


class ConverseDocumentPart(TypedDict):
    document: ConverseDocumentPartConfig


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
    ConverseDocumentPart,
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
    temperature: float
    topP: float
    maxTokens: int
    stopSequences: list[str]


class PerformanceConfig(TypedDict, total=False):
    latency: Literal["optimized", "standard"] | str


class ConverseRequest(TypedDict, total=False):
    messages: Required[list[ConverseMessage]]
    system: list[ConverseTextPart]
    inferenceConfig: InferenceConfig
    toolConfig: ConverseTools
    performanceConfig: PerformanceConfig


@dataclass
class ConverseRequestWrapper:
    messages: ListProjection[ConverseMessage]
    system: list[ConverseTextPart] | None = None
    inferenceConfig: InferenceConfig | None = None
    toolConfig: ConverseTools | None = None
    performanceConfig: PerformanceConfig | None = None

    def to_request(self) -> ConverseRequest:
        return ConverseRequest(
            messages=self.messages.raw_list,
            **remove_nones(
                {
                    "inferenceConfig": self.inferenceConfig,
                    "toolConfig": self.toolConfig,
                    "system": self.system,
                    "performanceConfig": self.performanceConfig,
                }
            ),
        )


ConverseDeployment = str


class ConverseImageType(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"

    @classmethod
    def all(cls) -> list["ConverseImageType"]:
        return list(cls)


class ConverseDocumentType(str, Enum):
    PDF = "pdf"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    XLS = "xls"
    XLSX = "xlsx"
    TXT = "txt"
    MD = "md"

    @classmethod
    def all(cls) -> list["ConverseDocumentType"]:
        return list(cls)
