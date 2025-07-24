import json
from typing import (
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    assert_never,
    cast,
    overload,
)

from aidial_sdk.chat_completion import (
    FinishReason,
    MessageContentImagePart,
    MessageContentTextPart,
    Tool,
    ToolCall,
)
from aidial_sdk.chat_completion.request import MessageContentRefusalPart
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControlEphemeralParam,
)
from anthropic.types.beta import BetaContentBlock as ContentBlock
from anthropic.types.beta import BetaContentBlockParam as ContentBlockParam
from anthropic.types.beta import BetaImageBlockParam as ImageBlockParam
from anthropic.types.beta import BetaMessageParam as MessageParam
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import BetaToolParam as ToolParam
from anthropic.types.beta import (
    BetaToolResultBlockParam as ToolResultBlockParam,
)
from anthropic.types.beta import BetaToolUseBlockParam as ToolUseBlockParam
from anthropic.types.beta import BetaUsage as Usage
from anthropic.types.beta.beta_base64_image_source_param import (
    BetaBase64ImageSourceParam as Base64ImageSourceParam,
)
from pydantic import BaseModel
from pydantic import ValidationError as PydValidationError

from aidial_adapter_bedrock.dial_api.resource import (
    AttachmentResource,
    DialResource,
    UnsupportedContentType,
    URLResource,
)
from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
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
from aidial_adapter_bedrock.utils.list import group_by
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
from aidial_adapter_bedrock.utils.resource import Resource

ClaudeFinishReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use"
]
ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
IMAGE_MEDIA_TYPES: List[str] = [
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
]

FILE_EXTENSIONS = ["png", "jpeg", "jpg", "gif", "webp"]


class MessageState(BaseModel):
    claude_message_content: List[ContentBlock]

    def to_dict(self) -> dict:
        return self.dict(
            # FIXME: a hack to exclude the private __json_buf field
            exclude={"claude_message_content": {"__all__": {"__json_buf"}}},
            # Excluding `citations: null`, since they could not be even parsed
            # currently by the Bedrock.
            exclude_none=True,
        )


def _get_message_content_from_state(
    idx: int, message: AIRegularMessage | AIToolCallMessage
) -> List[ContentBlockParam] | None:
    if (cc := message.custom_content) is not None and (
        state_dict := cc.state
    ) is not None:
        try:
            state = MessageState.parse_obj(state_dict)
            return [block.to_dict() for block in state.claude_message_content]  # type: ignore
        except PydValidationError as e:
            log.error(
                f"Invalid state at the path 'messages[{idx}].custom_content.state': {e}"
            )
            return None

    return None


def _create_text_block(text: str) -> TextBlockParam:
    return TextBlockParam(text=text, type="text")


def _create_image_block(resource: Resource) -> ImageBlockParam:
    return ImageBlockParam(
        source=Base64ImageSourceParam(
            data=resource.data_base64,
            media_type=cast(ImageMediaType, resource.type),
            type="base64",
        ),
        type="image",
    )


async def _collect_image_block(
    file_storage: FileStorage | None, dial_resource: DialResource
) -> ImageBlockParam:
    try:
        resource = await dial_resource.download(file_storage)
    except UnsupportedContentType as e:
        raise UserError(
            f"Unsupported media type: {e.type}",
            get_usage_message(FILE_EXTENSIONS),
        )

    return _create_image_block(resource)


_claude_cache_breakpoint = CacheControlEphemeralParam(type="ephemeral")


def _add_cache_control(
    message: BaseMessage | HumanToolResultMessage | AIToolCallMessage,
    claude_content: Iterable[ContentBlockParam],
) -> Iterable[ContentBlockParam]:

    if message.cache_breakpoint is not None:
        for block in reversed(list(claude_content)):
            if (
                block["type"] != "thinking"
                and block["type"] != "redacted_thinking"
            ):
                block["cache_control"] = _claude_cache_breakpoint
                break

    return claude_content


def _to_message_param(
    dial_message: (
        AIRegularMessage
        | AIToolCallMessage
        | HumanRegularMessage
        | HumanToolResultMessage
    ),
    claude_content: Iterable[ContentBlockParam],
) -> MessageParam:
    match dial_message:
        case AIRegularMessage() | AIToolCallMessage():
            role = "assistant"
        case HumanRegularMessage() | HumanToolResultMessage():
            role = "user"
        case _:
            assert_never(dial_message)
    return MessageParam(role=role, content=claude_content)


@overload
async def _to_claude_message(
    file_storage: FileStorage | None, message: SystemMessage
) -> Iterable[TextBlockParam]: ...


@overload
async def _to_claude_message(
    file_storage: FileStorage | None,
    message: AIRegularMessage | HumanRegularMessage,
) -> Iterable[TextBlockParam | ImageBlockParam]: ...


async def _to_claude_message(
    file_storage: FileStorage | None,
    message: SystemMessage | AIRegularMessage | HumanRegularMessage,
) -> Iterable[TextBlockParam | ImageBlockParam]:
    ret: List[TextBlockParam | ImageBlockParam] = []

    if isinstance(message, (AIRegularMessage, HumanRegularMessage)):
        for attachment in message.attachments:
            dial_resource = AttachmentResource(
                attachment=attachment,
                entity_name="image attachment",
                supported_types=IMAGE_MEDIA_TYPES,
            )
            ret.append(await _collect_image_block(file_storage, dial_resource))

    content = message.content

    match content:
        case str():
            if content:
                ret.append(_create_text_block(content))
        case list():
            for part in content:
                match part:
                    case MessageContentTextPart(text=text):
                        if text:
                            ret.append(_create_text_block(text))
                    case MessageContentImagePart(image_url=image_url):
                        dial_resource = URLResource(
                            url=image_url.url,
                            entity_name="image url",
                            supported_types=IMAGE_MEDIA_TYPES,
                        )
                        ret.append(
                            await _collect_image_block(
                                file_storage, dial_resource
                            )
                        )
                    case MessageContentRefusalPart():
                        raise ValidationError(
                            "Refusal message aren't supported"
                        )
                    case _:
                        assert_never(part)
        case _:
            assert_never(content)

    if not ret:
        ret.append(_create_text_block(""))

    return ret


def _to_claude_tool_call(call: ToolCall) -> ContentBlockParam:
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


def _merge_messages_with_same_role(
    messages: ListProjection[MessageParam],
) -> ListProjection[MessageParam]:
    def _key(message: Tuple[MessageParam, Set[int]]) -> str:
        return message[0]["role"]

    def _merge(
        a: Tuple[MessageParam, Set[int]],
        b: Tuple[MessageParam, Set[int]],
    ) -> Tuple[MessageParam, Set[int]]:
        (msg1, set1), (msg2, set2) = a, b

        content1 = msg1["content"]
        content2 = msg2["content"]

        if isinstance(content1, str):
            content1 = [TextBlockParam(type="text", text=content1)]

        if isinstance(content2, str):
            content2 = [TextBlockParam(type="text", text=content2)]

        return {
            "role": msg1["role"],
            "content": list(content1) + list(content2),
        }, set1 | set2

    return ListProjection(group_by(messages.list, _key, lambda x: x, _merge))


async def to_claude_messages(
    messages: List[BaseMessage | HumanToolResultMessage | AIToolCallMessage],
    file_storage: Optional[FileStorage],
) -> Tuple[List[TextBlockParam], ListProjection[MessageParam]]:

    idx_offset: int = 0
    system_messages: List[TextBlockParam] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            idx_offset += 1
            content = await _to_claude_message(file_storage, message)
            content = _add_cache_control(message, content)
            system_messages.extend(content)  # type: ignore
        else:
            break

    ret: ListProjection[MessageParam] = ListProjection()
    for idx, message in enumerate(messages[idx_offset:], start=idx_offset):

        match message:
            case HumanRegularMessage():
                content = await _to_claude_message(file_storage, message)

            case AIRegularMessage():
                # Take the message content from the state if possible,
                # since it may include certain content blocks that
                # are missing from the DIAL message itself,
                # such as thinking signatures and redacted thinking blocks.
                content = _get_message_content_from_state(idx, message)
                if content is None:
                    content = await _to_claude_message(file_storage, message)

            case AIToolCallMessage():
                content = _get_message_content_from_state(idx, message)

                if content is None:
                    content = [
                        _to_claude_tool_call(call) for call in message.calls
                    ]
                    if message.content is not None:
                        content.insert(0, _create_text_block(message.content))

            case HumanToolResultMessage():
                content = [_to_claude_tool_result(message)]

            case SystemMessage():
                raise ValidationError(
                    "System and developer messages are only allowed in the begging of the conversation."
                )
            case _:
                assert_never(message)

        claude_message = _to_message_param(
            message, _add_cache_control(message, content)
        )
        ret.append(claude_message, idx)

    return system_messages, _merge_messages_with_same_role(ret)


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


def to_dial_usage(usage: Usage) -> TokenUsage:
    read = usage.cache_creation_input_tokens or 0
    write = usage.cache_read_input_tokens or 0
    return TokenUsage(
        completion_tokens=usage.output_tokens,
        prompt_tokens=usage.input_tokens + read + write,
        cache_write_input_tokens=read,
        cache_read_input_tokens=write,
    )


def to_claude_tool_config(tool: Tool) -> ToolParam:
    function = tool.function
    tool_param = ToolParam(
        input_schema=function.parameters
        or {"type": "object", "properties": {}},
        name=function.name,
        description=function.description or "",
    )

    if tool.custom_fields and tool.custom_fields.cache_breakpoint:
        tool_param["cache_control"] = _claude_cache_breakpoint

    return tool_param


def get_usage_message(supported_exts: List[str]) -> str:
    return f"""
The application answers queries about attached images.
Attach images and ask questions about them in the same message.

Supported image types: {', '.join(supported_exts)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()
