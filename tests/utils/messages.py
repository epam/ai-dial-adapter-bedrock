from typing import List

from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    BaseMessage,
    HumanRegularMessage,
    SystemMessage,
    ToolMessage,
)


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIRegularMessage:
    return AIRegularMessage(content=content)


def user(content: str) -> HumanRegularMessage:
    return HumanRegularMessage(content=content)


def to_sdk_messages(messages: List[BaseMessage | ToolMessage]) -> List[Message]:
    return [msg.to_message() for msg in messages]
