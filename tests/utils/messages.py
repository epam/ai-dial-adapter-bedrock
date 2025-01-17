from aidial_sdk.chat_completion import Attachment, CustomContent, Message

from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    HumanRegularMessage,
    SystemMessage,
)


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
