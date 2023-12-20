from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    HumanRegularMessage,
    SystemMessage,
)


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIRegularMessage:
    return AIRegularMessage(content=content)


def user(content: str) -> HumanRegularMessage:
    return HumanRegularMessage(content=content)
