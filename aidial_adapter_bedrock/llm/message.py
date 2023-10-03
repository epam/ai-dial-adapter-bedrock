from abc import abstractmethod

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.exceptions import ValidationError


class BaseMessage(BaseModel):
    content: str

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""


class SystemMessage(BaseMessage):
    @property
    def type(self) -> str:
        return "system"


class HumanMessage(BaseMessage):
    @property
    def type(self) -> str:
        return "human"


class AIMessage(BaseMessage):
    @property
    def type(self) -> str:
        return "ai"


def parse_message(msg: Message) -> BaseMessage:
    if msg.content is None:
        raise ValidationError("Message content must be present")

    match msg.role:
        case Role.SYSTEM:
            return SystemMessage(content=msg.content)
        case Role.USER:
            return HumanMessage(content=msg.content)
        case Role.ASSISTANT:
            return AIMessage(content=msg.content)
        case Role.FUNCTION:
            raise ValidationError("Function calls are not supported")
