import mimetypes
from typing import Iterable, List, Literal, Optional, Tuple, assert_never, cast

from aidial_sdk.chat_completion import Attachment, FinishReason
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from anthropic.types.image_block_param import Source

from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    BaseMessage,
    HumanRegularMessage,
    SystemMessage,
)

ClaudeFinishReason = Literal["end_turn", "max_tokens", "stop_sequence"]
ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
IMAGE_MEDIA_TYPES: Iterable[ImageMediaType] = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}

FILE_EXTENSIONS = ["png", "jpeg", "jpg", "gif", "webp"]


def _validate_media_type(media_type: str) -> ImageMediaType:
    if media_type not in IMAGE_MEDIA_TYPES:
        raise UserError(
            f"Unsupported media type: {media_type}",
            get_usage_message(FILE_EXTENSIONS),
        )
    return cast(ImageMediaType, media_type)


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


def _to_claude_role(
    message: BaseMessage,
) -> Tuple[
    Literal["user", "assistant"], AIRegularMessage | HumanRegularMessage
]:
    match message:
        case HumanRegularMessage():
            return "user", message
        case AIRegularMessage():
            return "assistant", message
        case SystemMessage():
            raise ValueError(
                "System message is only allowed as the first message"
            )
        case _:
            assert_never(message)


async def _to_claude_image(
    attachment: Attachment, file_storage: Optional[FileStorage]
) -> ImageBlockParam:
    if attachment.data:
        if not attachment.type:
            raise ValidationError(
                "Attachment type is required for provided data"
            )
        return _create_image_block(
            _validate_media_type(attachment.type), attachment.data
        )

    if attachment.url:
        media_type = attachment.type or mimetypes.guess_type(attachment.url)[0]
        if not media_type:
            raise ValidationError(
                f"Cannot guess attachment type for {attachment.url}"
            )

        data = await _download_data(attachment.url, file_storage)
        return _create_image_block(_validate_media_type(media_type), data)

    raise ValidationError("Attachment data or URL is required")


async def _to_claude_content(
    message: AIRegularMessage | HumanRegularMessage,
    file_storage: Optional[FileStorage],
) -> List[TextBlockParam | ImageBlockParam]:
    content: List[TextBlockParam | ImageBlockParam] = []

    if message.custom_content:
        for attachment in message.custom_content.attachments or []:
            content.append(await _to_claude_image(attachment, file_storage))

    content.append(TextBlockParam(text=message.content, type="text"))
    return content


async def to_claude_messages(
    messages: List[BaseMessage], file_storage: Optional[FileStorage]
) -> Tuple[Optional[str], List[MessageParam]]:
    if not messages:
        return None, []

    system_prompt: str | None = None
    if isinstance(messages[0], SystemMessage):
        system_prompt = messages[0].content
        messages = messages[1:]

    claude_messages: List[MessageParam] = []
    for message in messages:
        role, message = _to_claude_role(message)
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
        case _:
            assert_never(finish_reason)


def get_usage_message(supported_exts: List[str]) -> str:
    return f"""
The application answers queries about attached images.
Attach images and ask questions about them in the same message.

Supported image types: {', '.join(supported_exts)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()