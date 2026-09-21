from aidial_adapter_anthropic.dial._message import (
    AIRegularMessage,
    HumanRegularMessage,
    SystemMessage,
    parse_dial_message,
)
from aidial_adapter_anthropic.dial.request import AdapterRequest
from aidial_sdk.chat_completion import Attachment, CustomContent, Message

from aidial_adapter_bedrock.utils.list_projection import ListProjection


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


def adapter_request(
    messages: list[Message] | None = None, **kwargs
) -> AdapterRequest:
    """Builds an `AdapterRequest` the way `AdapterRequest.create` would.

    The adapters take the messages and the parameters as one object now, so a
    test that only cares about the parameters still has to supply messages.
    """
    return AdapterRequest(
        messages=ListProjection.create(
            [parse_dial_message(msg) for msg in messages or []]
        ),
        **kwargs,
    )
