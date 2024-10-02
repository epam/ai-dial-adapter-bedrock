from typing import List, Optional, TypeGuard

from aidial_sdk.chat_completion import (
    MessageContentPart,
    MessageContentTextPart,
)
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.tools.tools_config import (
    ToolsConfig,
    ToolsMode,
    validate_messages,
)


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

    @property
    def tools_mode(self) -> ToolsMode | None:
        if self.tool_config is not None:
            return self.tool_config.tools_mode
        return None


def collect_text_content(
    content: str | List[MessageContentPart] | None,
    delimiter: str = "\n",
    strict: bool = True,
) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    texts: List[str] = []
    for part in content:
        if isinstance(part, MessageContentTextPart):
            texts.append(part.text)
        elif strict:
            raise ValidationError(
                "Can't extract text from a multi-modal content part"
            )

    return delimiter.join(texts)


def is_text_content_parts(
    content: List[MessageContentPart],
) -> TypeGuard[List[MessageContentTextPart]]:
    return all(isinstance(part, MessageContentTextPart) for part in content)


def is_plain_text_content(
    content: str | List[MessageContentPart] | None,
) -> TypeGuard[str | None]:
    return content is None or isinstance(content, str)
