from typing import assert_never

from aidial_adapter_anthropic.adapter import ValidationError
from aidial_adapter_anthropic.dial.resource import (
    AttachmentResource,
    DialResource,
    URLResource,
)
from aidial_sdk.chat_completion import (
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
    Role,
)
from aidial_sdk.chat_completion.request import (
    ImageURL,
    MessageContentRefusalPart,
)


def parse_message(
    message: Message, supported_types: list[str]
) -> tuple[str | None, list[DialResource]]:
    text_prompt: str | None = None
    image_resources: list[DialResource] = []

    match message.content:
        case str(text):
            text_prompt = text
        case list():
            text_parts: list[str] = []

            for part in message.content:
                match part:
                    case MessageContentTextPart(text=text):
                        text_parts.append(text)
                    case MessageContentImagePart(image_url=ImageURL(url=url)):
                        image_resources.append(
                            URLResource(
                                url=url, supported_types=supported_types
                            )
                        )
                    case MessageContentRefusalPart():
                        raise ValidationError(
                            "Refusal messages aren't supported"
                        )
                    case _:
                        assert_never(part)
            if text_parts:
                text_prompt = " ".join(text_parts)
        case None:
            pass
        case _:
            assert_never(message.content)

    if message.custom_content and message.custom_content.attachments:
        image_resources.extend(
            [
                AttachmentResource(
                    attachment=attachment,
                    supported_types=supported_types,
                )
                for attachment in message.custom_content.attachments
            ]
        )

    return text_prompt, image_resources


def validate_last_message(messages: list[Message]):
    if not messages:
        raise ValidationError("No messages provided")

    last_message = messages[-1]
    if last_message.role != Role.USER:
        raise ValidationError("Last message must be from user")
    return last_message
