import aidial_sdk.chat_completion as sdk
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.exceptions import ValidationError


class BaseMessage(BaseModel):
    content: str


class SystemMessage(BaseMessage):
    pass


class HumanMessage(BaseMessage):
    pass


class AIMessage(BaseMessage):
    pass


def parse_message(msg: sdk.Message) -> BaseMessage:
    match msg:
        case sdk.SystemMessage(content=content):
            return SystemMessage(content=content)
        case sdk.UserMessage(content=content):
            return HumanMessage(content=content)
        case sdk.AssistantMessage(content=content):
            return AIMessage(content=content)
        case sdk.FunctionMessage():
            raise ValidationError("Function calls are not supported")
        case sdk.ToolMessage():
            raise ValidationError("Tool calls are not supported")
        case _:
            raise ValidationError("Unknown message type")
