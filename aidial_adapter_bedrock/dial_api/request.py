from enum import Enum
from typing import List, Literal, Optional

from aidial_sdk.chat_completion.request import ChatCompletionRequest
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.tools.tools_config import (
    ToolsConfig,
    validate_messages,
)
from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel


class ModelParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: List[str] = []
    max_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    stream: bool = False
    tool_config: Optional[ToolsConfig] = None

    @classmethod
    def create(cls, request: ChatCompletionRequest) -> "ModelParameters":
        stop: List[str] = []
        if request.stop is not None:
            stop = (
                [request.stop]
                if isinstance(request.stop, str)
                else request.stop
            )

        validate_messages(request)

        return cls(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=stop,
            max_tokens=request.max_tokens,
            max_prompt_tokens=request.max_prompt_tokens,
            stream=request.stream,
            tool_config=ToolsConfig.from_request(request),
        )

    def add_stop_sequences(self, stop: List[str]) -> "ModelParameters":
        return self.copy(update={"stop": [*self.stop, *stop]})


class EmbeddingsType(str, Enum):
    SYMMETRIC = "symmetric"
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingsRequest(ExtraAllowModel):
    input: str | List[str]
    user: Optional[str] = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None

    @staticmethod
    def example() -> "EmbeddingsRequest":
        return EmbeddingsRequest(input=["fish", "ball"])
