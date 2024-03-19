import mimetypes
from typing import Iterable, List, Literal, Optional, Tuple, Union

from aidial_sdk.chat_completion import Attachment, FinishReason, Message, Role
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from anthropic.types.image_block_param import Source

from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    download_file_as_base64,
)

ClaudeFinishReason = Literal["end_turn", "max_tokens", "stop_sequence"]
ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
IMAGE_MEDIA_TYPES: Iterable[ImageMediaType] = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}


def _validate_media_type(media_type: str) -> ImageMediaType:
    if media_type not in IMAGE_MEDIA_TYPES:
        raise ValueError(f"Unsupported media type: {media_type}")
    return media_type  # type: ignore


def _create_image_block(
    media_type: ImageMediaType, data: str
) -> ImageBlockParam:
    return ImageBlockParam(
        source=Source(
            data=data,
            media_type=media_type,
            type="base64",
        ),
        type="image",
    )


async def _download_data(url: str, file_storage: Optional[FileStorage]) -> str:
    if not file_storage:
        return await download_file_as_base64(url)

    return await file_storage.download_file_as_base64(url)


def _to_claude_role(role: Role) -> Literal["user", "assistant"]:
    match role:
        case Role.USER:
            return "user"
        case Role.ASSISTANT:
            return "assistant"
        case _:
            raise ValueError(f"Unsupported role: {role}")


async def _to_claude_image(
    attachment: Attachment, file_storage: Optional[FileStorage]
) -> ImageBlockParam:
    if attachment.data:
        if not attachment.type:
            raise ValueError("Attachment type is required for provided data")
        return _create_image_block(
            _validate_media_type(attachment.type), attachment.data
        )

    if attachment.url:
        media_type = attachment.type or mimetypes.guess_type(attachment.url)[0]
        if not media_type:
            raise ValueError(
                f"Cannot guess attachment type for {attachment.url}"
            )

        data = await _download_data(attachment.url, file_storage)
        return _create_image_block(_validate_media_type(media_type), data)

    raise ValueError("Attachment data or URL is required")


async def _to_claude_content(
    message: Message, file_storage: Optional[FileStorage]
) -> Union[str, List[Union[TextBlockParam, ImageBlockParam]]]:
    if message.custom_content and message.custom_content.attachments:
        content: List[Union[TextBlockParam, ImageBlockParam]] = []
        for attachment in message.custom_content.attachments:
            content.append(await _to_claude_image(attachment, file_storage))

        if message.content:
            content.append(TextBlockParam(text=message.content, type="text"))

        return content

    return message.content or ""


async def to_claude_messages(
    messages: List[Message], file_storage: Optional[FileStorage]
) -> Tuple[Optional[str], List[MessageParam]]:
    if not messages:
        return None, []

    system_prompt: str | None = None
    first_message_index = 0
    if messages[0].role == Role.SYSTEM:
        system_prompt = messages[0].content
        first_message_index = 1

    claude_messages: List[MessageParam] = []
    for i in range(first_message_index, len(messages)):
        message = messages[i]
        if message.role == Role.SYSTEM:
            raise ValueError(
                "System message is only allowed as the first message"
            )

        role = _to_claude_role(message.role)
        content = await _to_claude_content(message, file_storage)
        claude_messages.append(MessageParam(role=role, content=content))

    return system_prompt, claude_messages


def to_dial_finish_reason(
    finish_reason: Optional[ClaudeFinishReason],
) -> FinishReason:
    if finish_reason is None:
        return FinishReason.STOP

    match finish_reason:
        case "end_turn":
            return FinishReason.STOP
        case "max_tokens":
            return FinishReason.LENGTH
        case "stop_sequence":
            return FinishReason.STOP
