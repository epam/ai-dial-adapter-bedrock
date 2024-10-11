from typing import List, Optional, TypeGuard, assert_never

from aidial_sdk.chat_completion import (
    MessageContentImagePart,
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

MessageContent = str | List[MessageContentPart] | None
MessageContentSpecialized = (
    MessageContent
    | List[MessageContentTextPart]
    | List[MessageContentImagePart]
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
    content: MessageContentSpecialized, delimiter: str = "\n\n"
) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    texts: List[str] = []
    for part in content:
        if isinstance(part, MessageContentTextPart):
            texts.append(part.text)
        else:
            raise ValidationError(
                "Can't extract text from a multi-modal content part"
            )

    return delimiter.join(texts)


def to_message_content(content: MessageContentSpecialized) -> MessageContent:
    match content:
        case None | str():
            return content
        case list():
            return [*content]
        case _:
            assert_never(content)


def is_text_content_parts(
    content: List[MessageContentPart],
) -> TypeGuard[List[MessageContentTextPart]]:
    return all(isinstance(part, MessageContentTextPart) for part in content)


def is_plain_text_content(content: MessageContent) -> TypeGuard[str | None]:
    return content is None or isinstance(content, str)
