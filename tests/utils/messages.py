from typing import List

from aidial_sdk.chat_completion import Attachment, CustomContent
from aidial_sdk.chat_completion import Message as DialMessage

from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    HumanRegularMessage,
    MessageABC,
    SystemMessage,
)


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIRegularMessage:
    return AIRegularMessage(content=content)


def user(content: str) -> HumanRegularMessage:
    return HumanRegularMessage(content=content)


def user_with_image(content: str, image_base64: str) -> HumanRegularMessage:
    custom_content = CustomContent(
        attachments=[Attachment(type="image/png", data=image_base64)]
    )
    return HumanRegularMessage(content=content, custom_content=custom_content)


def to_sdk_messages(messages: List[MessageABC]) -> List[DialMessage]:
    return [msg.to_message() for msg in messages]
