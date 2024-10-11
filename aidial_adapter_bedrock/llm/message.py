from abc import ABC, abstractmethod
from typing import List, Optional, Self, Union

from aidial_sdk.chat_completion import Attachment, CustomContent, FunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import (
    MessageContentPart,
    MessageContentTextPart,
    Role,
    ToolCall,
)
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.request import (
    collect_text_content,
    is_plain_text_content,
    is_text_content,
    to_message_content,
)
from aidial_adapter_bedrock.llm.errors import ValidationError


class MessageABC(ABC, BaseModel):
    @abstractmethod
    def to_message(self) -> DialMessage: ...

    @classmethod
    @abstractmethod
    def from_message(cls, message: DialMessage) -> Self | None: ...


class BaseMessageABC(MessageABC):
    @property
    @abstractmethod
    def text_content(self) -> str: ...


class SystemMessage(BaseMessageABC):
    content: str | List[MessageContentTextPart]

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.SYSTEM,
            content=to_message_content(self.content),
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.SYSTEM:
            return None

        content = message.content

        if not is_text_content(content):
            raise ValidationError(
                "System message is expected to be a string or a list of text content parts"
            )

        return cls(content=content)

    @property
    def text_content(self) -> str:
        return collect_text_content(self.content)


class HumanRegularMessage(BaseMessageABC):
    """MM stands for multi-modal"""

    content: str | List[MessageContentPart]
    custom_content: Optional[CustomContent] = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.USER,
            content=self.content,
            custom_content=self.custom_content,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.USER:
            return None

        content = message.content
        if content is None:
            raise ValidationError(
                "User message is expected to have content field"
            )

        return cls(content=content, custom_content=message.custom_content)

    @property
    def text_content(self) -> str:
        return collect_text_content(self.content)

    @property
    def attachments(self) -> List[Attachment]:
        return (
            self.custom_content.attachments or [] if self.custom_content else []
        )


class HumanToolResultMessage(MessageABC):
    id: str
    content: str

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.TOOL,
            tool_call_id=self.id,
            content=self.content,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.TOOL:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The tool message shouldn't contain content parts"
            )

        if message.content is None or message.tool_call_id is None:
            raise ValidationError(
                "The tool message is expected to have content and tool_call_id fields"
            )

        return cls(id=message.tool_call_id, content=message.content)


class HumanFunctionResultMessage(MessageABC):
    name: str
    content: str

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.FUNCTION,
            name=self.name,
            content=self.content,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.FUNCTION:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The function message shouldn't contain content parts"
            )

        if message.content is None or message.name is None:
            raise ValidationError(
                "The function message is expected to have content and name fields"
            )

        return cls(name=message.name, content=message.content)


class AIRegularMessage(BaseMessageABC):
    content: str
    custom_content: Optional[CustomContent] = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.ASSISTANT,
            content=self.content,
            custom_content=self.custom_content,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.ASSISTANT:
            return None

        if message.function_call is not None or message.tool_calls is not None:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The assistant message shouldn't contain content parts"
            )

        if message.content is None:
            raise ValidationError(
                "The assistant message is expected to have content"
            )

        return cls(
            content=message.content, custom_content=message.custom_content
        )

    @property
    def text_content(self) -> str:
        return self.content

    @property
    def attachments(self) -> List[Attachment]:
        return (
            self.custom_content.attachments or [] if self.custom_content else []
        )


class AIToolCallMessage(MessageABC):
    calls: List[ToolCall]
    content: Optional[str] = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.ASSISTANT,
            content=self.content,
            tool_calls=self.calls,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.ASSISTANT:
            return None

        if message.tool_calls is None or message.function_call is not None:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The assistant message with tool calls shouldn't contain content parts"
            )

        return cls(calls=message.tool_calls, content=message.content)


class AIFunctionCallMessage(MessageABC):
    call: FunctionCall
    content: Optional[str] = None

    def to_message(self) -> DialMessage:
        return DialMessage(
            role=Role.ASSISTANT,
            content=self.content,
            function_call=self.call,
        )

    @classmethod
    def from_message(cls, message: DialMessage) -> Self | None:
        if message.role != Role.ASSISTANT:
            return None

        if message.function_call is None or message.tool_calls is not None:
            return None

        if not is_plain_text_content(message.content):
            raise ValidationError(
                "The assistant message with function call shouldn't contain content parts"
            )

        return cls(call=message.function_call, content=message.content)


BaseMessage = Union[SystemMessage, HumanRegularMessage, AIRegularMessage]

ToolMessage = Union[
    HumanToolResultMessage,
    HumanFunctionResultMessage,
    AIToolCallMessage,
    AIFunctionCallMessage,
]


def parse_dial_message(msg: DialMessage) -> BaseMessage | ToolMessage:

    message = (
        SystemMessage.from_message(msg)
        or HumanRegularMessage.from_message(msg)
        or HumanToolResultMessage.from_message(msg)
        or HumanFunctionResultMessage.from_message(msg)
        or AIRegularMessage.from_message(msg)
        or AIToolCallMessage.from_message(msg)
        or AIFunctionCallMessage.from_message(msg)
    )

    if message is None:
        raise ValidationError("Unknown message type or invalid message")

    return message
