from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.dial._message import (
    AdapterMessage,
    AIRegularMessage,
    HumanRegularMessage,
    SystemMessage,
    parse_dial_message,
)
from aidial_sdk.chat_completion import Attachment, CustomContent, Message


def sys(content: str) -> Message:
    return SystemMessage(content=content).to_message()


def ai(content: str) -> Message:
    return AIRegularMessage(content=content).to_message()


def user(content: str) -> Message:
    return HumanRegularMessage(content=content).to_message()


def user_with_image(content: str, image_base64: str) -> Message:
    custom_content = CustomContent(
        attachments=[Attachment(type="image/png", data=image_base64)]
    )
    return HumanRegularMessage(
        content=content, custom_content=custom_content
    ).to_message()


def parse_messages(messages: list[Message]) -> ListProjection[AdapterMessage]:
    return ListProjection.create([parse_dial_message(msg) for msg in messages])
