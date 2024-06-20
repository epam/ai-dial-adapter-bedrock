import json
import mimetypes
from typing import List, Literal, Optional, Set, Tuple, assert_never, cast

from aidial_sdk.chat_completion import (
    Attachment,
    FinishReason,
    Function,
    ToolCall,
)
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source

from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    AIToolCallMessage,
    BaseMessage,
    HumanRegularMessage,
    HumanToolResultMessage,
    SystemMessage,
    ToolMessage,
)
from aidial_adapter_bedrock.llm.model.claude.v3.tools import ToolsMode

ClaudeFinishReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use"
]
ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
IMAGE_MEDIA_TYPES: Set[ImageMediaType] = {
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


async def _basic_message_to_claude_content(
    message: AIRegularMessage | HumanRegularMessage,
    file_storage: Optional[FileStorage],
) -> List[TextBlockParam | ImageBlockParam]:
    content: List[TextBlockParam | ImageBlockParam] = []

    if message.custom_content:
        for attachment in message.custom_content.attachments or []:
            content.append(await _to_claude_image(attachment, file_storage))

    content.append(TextBlockParam(text=message.content, type="text"))
    return content


def _tool_call_to_claude_content(call: ToolCall) -> ToolUseBlockParam:
    return ToolUseBlockParam(
        id=call.id,
        name=call.function.name,
        input=json.loads(call.function.arguments),
        type="tool_use",
    )


def _tool_result_to_claude_content(
    message: HumanToolResultMessage,
) -> ToolResultBlockParam:
    return ToolResultBlockParam(
        tool_use_id=message.id,
        type="tool_result",
        content=[TextBlockParam(text=message.content, type="text")],
    )


async def to_claude_messages(
    messages: List[BaseMessage | ToolMessage],
    file_storage: Optional[FileStorage],
) -> Tuple[Optional[str], List[MessageParam]]:
    if not messages:
        return None, []

    system_prompt: str | None = None
    if isinstance(messages[0], SystemMessage):
        system_prompt = messages[0].content
        messages = messages[1:]

    claude_messages: List[MessageParam] = []
    for message in messages:
        match message:
            case HumanRegularMessage():
                claude_messages.append(
                    MessageParam(
                        role="user",
                        content=await _basic_message_to_claude_content(
                            message, file_storage
                        ),
                    )
                )
            case AIRegularMessage():
                claude_messages.append(
                    MessageParam(
                        role="assistant",
                        content=await _basic_message_to_claude_content(
                            message, file_storage
                        ),
                    )
                )
            case AIToolCallMessage():
                claude_messages.append(
                    MessageParam(
                        role="assistant",
                        content=[
                            _tool_call_to_claude_content(call)
                            for call in message.calls
                        ],
                    )
                )
            case HumanToolResultMessage():
                claude_messages.append(
                    MessageParam(
                        role="user",
                        content=[_tool_result_to_claude_content(message)],
                    )
                )
            case SystemMessage():
                raise ValueError(
                    "System message is only allowed as the first message"
                )
            case _:
                raise ValidationError(
                    f"Unknown type of of message! {type(message)}"
                )

    return system_prompt, claude_messages


def to_dial_finish_reason(
    finish_reason: Optional[ClaudeFinishReason],
    tools_mode: ToolsMode | None,
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
        case "tool_use":
            match tools_mode:
                case ToolsMode.NATIVE_TOOLS:
                    return FinishReason.TOOL_CALLS
                case ToolsMode.FUNCTION_EMULATION:
                    return FinishReason.FUNCTION_CALL
                case _:
                    raise Exception(
                        f"Invalid tools mode {tools_mode} during tool use!"
                    )
        case _:
            assert_never(finish_reason)


def to_claude_tool_config(function_call: Function) -> ToolParam:
    return ToolParam(
        input_schema=function_call.parameters,
        name=function_call.name,
        description=function_call.description or "",
    )


def get_usage_message(supported_exts: List[str]) -> str:
    return f"""
The application answers queries about attached images.
Attach images and ask questions about them in the same message.

Supported image types: {', '.join(supported_exts)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()
