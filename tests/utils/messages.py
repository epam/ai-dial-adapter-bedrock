from aidial_adapter_anthropic._utils.list import ListProjection
from aidial_adapter_anthropic.dial._message import parse_dial_message
from aidial_adapter_anthropic.dial.request import AdapterRequest
from aidial_sdk.chat_completion import Attachment, CustomContent, Message

from aidial_adapter_bedrock.llm.chat_model import AdapterMessages
from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    HumanRegularMessage,
    SystemMessage,
)


def to_adapter_messages(messages: list[Message]) -> AdapterMessages:
    return ListProjection.create([parse_dial_message(msg) for msg in messages])


def adapter_request(
    messages: list[Message] | None = None, **kwargs
) -> AdapterRequest:
    """Builds an `AdapterRequest` the way `AdapterRequest.create` would.

    The adapters take the messages and the parameters as one object now, so a
    test that only cares about the parameters still has to supply messages.
    """
    return AdapterRequest(
        messages=to_adapter_messages(messages or []), **kwargs
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
