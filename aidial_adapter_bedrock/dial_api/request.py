from typing import List, Optional, Type, TypeGuard, TypeVar, assert_never

from aidial_sdk.chat_completion import (
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
)
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.exceptions import RequestValidationError
from aidial_sdk.pydantic_v1 import ValidationError as PydanticValidationError
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

_Model = TypeVar("_Model", bound=BaseModel)


class ModelParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stop: List[str] = []
    max_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    stream: bool = False
    tool_config: Optional[ToolsConfig] = None
    configuration: Optional[dict] = None

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

        configuration = (
            cf.configuration
            if (cf := request.custom_fields) is not None
            else None
        )

        return cls(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n or 1,
            stop=stop,
            max_tokens=request.max_tokens,
            max_prompt_tokens=request.max_prompt_tokens,
            stream=request.stream,
            tool_config=ToolsConfig.from_request(request),
            configuration=configuration,
        )

    def add_stop_sequences(self, stop: List[str]) -> "ModelParameters":
        return self.copy(update={"stop": [*self.stop, *stop]})

    @property
    def tools_mode(self) -> ToolsMode | None:
        if self.tool_config is not None:
            return self.tool_config.tools_mode
        return None

    def parse_configuration(self, cls: Type[_Model]) -> _Model | None:
        try:
            return cls.parse_obj(self.configuration or {})
        except PydanticValidationError as e:
            error = e.errors()[0]
            path = ".".join(map(str, error["loc"]))
            msg = f"Invalid request. Path: 'custom_field.configuration.{path}', error: {error['msg']}"

            raise RequestValidationError(msg)


def collect_text_content(
    content: MessageContentSpecialized, delimiter: str = "\n\n"
) -> str:
    match content:
        case None:
            return ""
        case str():
            return content
        case list():
            texts: List[str] = []
            for part in content:
                match part:
                    case MessageContentTextPart(text=text):
                        texts.append(text)
                    case MessageContentImagePart():
                        raise ValidationError(
                            "Can't extract text from an image content part"
                        )
                    case _:
                        assert_never(part)
            return delimiter.join(texts)
        case _:
            assert_never(content)


def to_message_content(content: MessageContentSpecialized) -> MessageContent:
    match content:
        case None | str():
            return content
        case list():
            return [*content]
        case _:
            assert_never(content)


def is_text_content(
    content: MessageContent,
) -> TypeGuard[str | List[MessageContentTextPart]]:
    match content:
        case None:
            return False
        case str():
            return True
        case list():
            return all(
                isinstance(part, MessageContentTextPart) for part in content
            )
        case _:
            assert_never(content)


def is_plain_text_content(content: MessageContent) -> TypeGuard[str | None]:
    return content is None or isinstance(content, str)
