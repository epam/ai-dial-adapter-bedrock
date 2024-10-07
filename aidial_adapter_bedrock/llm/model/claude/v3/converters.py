import json
from typing import List, Literal, Optional, Set, Tuple, assert_never, cast

from aidial_sdk.chat_completion import (
    FinishReason,
    Function,
    MessageContentImagePart,
    MessageContentTextPart,
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

from aidial_adapter_bedrock.dial_api.attachment import (
    download_attachment,
    download_url,
)
from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    AIToolCallMessage,
    BaseMessage,
    HumanRegularMessage,
    HumanToolResultMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode
from aidial_adapter_bedrock.utils.resource import Resource

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


def _create_text_block(text: str) -> TextBlockParam:
    return TextBlockParam(text=text, type="text")


def _create_image_block(resource: Resource) -> ImageBlockParam:
    return ImageBlockParam(
        source=Source(
            data=resource.data_base64,
            media_type=_validate_media_type(resource.type),
            type="base64",
        ),
        type="image",
    )


async def _to_claude_message(
    file_storage: Optional[FileStorage],
    message: AIRegularMessage | HumanRegularMessage,
) -> List[TextBlockParam | ImageBlockParam]:
    ret: List[TextBlockParam | ImageBlockParam] = []

    for attachment in message.attachments:
        resource = await download_attachment(
            file_storage, "image attachment", attachment
        )
        ret.append(_create_image_block(resource))

    content = message.content

    match content:
        case str():
            ret.append(_create_text_block(content))
        case list():
            for part in content:
                match part:
                    case MessageContentTextPart(text=text):
                        ret.append(_create_text_block(text))
                    case MessageContentImagePart(image_url=image_url):
                        resource = await download_url(
                            file_storage, "image url", image_url.url
                        )
                        ret.append(_create_image_block(resource))
                    case _:
                        assert_never(part)
        case _:
            assert_never(content)

    return ret


def _to_claude_tool_call(call: ToolCall) -> ToolUseBlockParam:
    return ToolUseBlockParam(
        id=call.id,
        name=call.function.name,
        input=json.loads(call.function.arguments),
        type="tool_use",
    )


def _to_claude_tool_result(
    message: HumanToolResultMessage,
) -> ToolResultBlockParam:
    return ToolResultBlockParam(
        tool_use_id=message.id,
        type="tool_result",
        content=[_create_text_block(message.content)],
    )


async def to_claude_messages(
    messages: List[BaseMessage | HumanToolResultMessage | AIToolCallMessage],
    file_storage: Optional[FileStorage],
) -> Tuple[Optional[str], List[MessageParam]]:
    if not messages:
        return None, []

    system_prompt: str | None = None
    if isinstance(messages[0], SystemMessage):
        system_prompt = messages[0].text_content
        messages = messages[1:]

    claude_messages: List[MessageParam] = []
    for message in messages:
        match message:
            case HumanRegularMessage():
                claude_messages.append(
                    MessageParam(
                        role="user",
                        content=await _to_claude_message(file_storage, message),
                    )
                )
            case AIRegularMessage():
                claude_messages.append(
                    MessageParam(
                        role="assistant",
                        content=await _to_claude_message(file_storage, message),
                    )
                )
            case AIToolCallMessage():
                content: List[TextBlockParam | ToolUseBlockParam] = [
                    _to_claude_tool_call(call) for call in message.calls
                ]
                if message.content is not None:
                    content.insert(0, _create_text_block(message.content))

                claude_messages.append(
                    MessageParam(
                        role="assistant",
                        content=content,
                    )
                )
            case HumanToolResultMessage():
                claude_messages.append(
                    MessageParam(
                        role="user",
                        content=[_to_claude_tool_result(message)],
                    )
                )
            case SystemMessage():
                raise ValidationError(
                    "System message is only allowed as the first message"
                )
            case _:
                assert_never(message)

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
                case ToolsMode.TOOLS:
                    return FinishReason.TOOL_CALLS
                case ToolsMode.FUNCTIONS:
                    return FinishReason.FUNCTION_CALL
                case None:
                    raise ValidationError(
                        "A model has called a tool, but no tools were given to the model in the first place."
                    )
                case _:
                    assert_never(tools_mode)

        case _:
            assert_never(finish_reason)


def to_claude_tool_config(function_call: Function) -> ToolParam:
    return ToolParam(
        input_schema=function_call.parameters
        or {"type": "object", "properties": {}},
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
