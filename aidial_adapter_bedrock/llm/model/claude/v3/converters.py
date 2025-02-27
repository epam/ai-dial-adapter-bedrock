import json
from typing import List, Literal, Optional, Set, Tuple, assert_never, cast

from aidial_sdk.chat_completion import FinishReason, Function, ToolCall
from anthropic.types import (
    Base64PDFSourceParam,
    CitationsConfigParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    PlainTextSourceParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.image_block_param import Source

from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    AIToolCallMessage,
    BaseMessage,
    HumanRegularMessage,
    HumanToolResultMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.llm.model.attachment_processor import (
    AttachmentProcessor,
    AttachmentProcessors,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode
from aidial_adapter_bedrock.utils.concurrency import aiter_to_list
from aidial_adapter_bedrock.utils.list import group_by
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from aidial_adapter_bedrock.utils.resource import Resource

_ClaudeFinishReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use"
]

_ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]


def _create_text_block(text: str) -> TextBlockParam:
    return TextBlockParam(text=text, type="text")


def _create_image_block(resource: Resource) -> ImageBlockParam:
    return ImageBlockParam(
        source=Source(
            data=resource.data_base64,
            media_type=cast(_ImageMediaType, resource.type),
            type="base64",
        ),
        type="image",
    )


_CITATIONS_ENABLED = True


def _create_text_document_block(resource: Resource) -> DocumentBlockParam:
    return DocumentBlockParam(
        source=PlainTextSourceParam(
            data=resource.data.decode("utf-8"),
            media_type=resource.type,  # type: ignore
            type="text",
        ),
        type="document",
        citations=CitationsConfigParam(enabled=_CITATIONS_ENABLED),
        title="",  # FIXME
    )


def _create_pdf_document_block(resource: Resource) -> DocumentBlockParam:
    return DocumentBlockParam(
        source=Base64PDFSourceParam(
            data=resource.data_base64,
            media_type=resource.type,  # type: ignore
            type="base64",
        ),
        type="document",
        citations=CitationsConfigParam(enabled=_CITATIONS_ENABLED),
        title="",  # FIXME
    )


IMAGE_ATTACHMENT_PROCESSOR = AttachmentProcessor(
    supported_types={
        "image/png": {"png"},
        "image/jpeg": {"jpeg", "jpg"},
        "image/gif": {"gif"},
        "image/webp": {"webp"},
    },
    handler=_create_image_block,
)

PDF_ATTACHMENT_PROCESSOR = AttachmentProcessor(
    supported_types={"application/pdf": {"pdf"}},
    handler=_create_pdf_document_block,
)

TEXT_ATTACHMENT_PROCESSOR = AttachmentProcessor(
    supported_types={"text/plain": {"txt"}},
    handler=_create_text_document_block,
)


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
            content1 = [
                cast(TextBlockParam, {"type": "text", "text": content1})
            ]

        if isinstance(content2, str):
            content2 = [
                cast(TextBlockParam, {"type": "text", "text": content2})
            ]

        return {
            "role": msg1["role"],
            "content": list(content1) + list(content2),
        }, set1 | set2

    return ListProjection(group_by(messages.list, _key, lambda x: x, _merge))


async def to_claude_messages(
    handlers: AttachmentProcessors[
        TextBlockParam | ImageBlockParam | DocumentBlockParam
    ],
    messages: List[BaseMessage | HumanToolResultMessage | AIToolCallMessage],
) -> Tuple[Optional[str], ListProjection[MessageParam]]:

    system_prompt: str | None = None
    if messages and isinstance(messages[0], SystemMessage):
        system_prompt = messages[0].text_content
        messages = messages[1:]

    idx_offset = int(system_prompt is not None)

    ret: ListProjection[MessageParam] = ListProjection()
    for idx, message in enumerate(messages, start=idx_offset):
        match message:
            case HumanRegularMessage() | AIRegularMessage():
                role = (
                    "user"
                    if isinstance(message, HumanRegularMessage)
                    else "assistant"
                )
                blocks = handlers.process_attachments(
                    _create_text_block, message
                )
                ret.append(
                    MessageParam(
                        role=role, content=await aiter_to_list(blocks)
                    ),
                    idx,
                )
            case AIToolCallMessage():
                content: List[TextBlockParam | ToolUseBlockParam] = [
                    _to_claude_tool_call(call) for call in message.calls
                ]
                if message.content is not None:
                    content.insert(0, _create_text_block(message.content))

                ret.append(
                    MessageParam(
                        role="assistant",
                        content=content,
                    ),
                    idx,
                )
            case HumanToolResultMessage():
                ret.append(
                    MessageParam(
                        role="user",
                        content=[_to_claude_tool_result(message)],
                    ),
                    idx,
                )
            case SystemMessage():
                raise ValidationError(
                    "System message is only allowed as the first message"
                )
            case _:
                assert_never(message)

    return system_prompt, _merge_messages_with_same_role(ret)


def to_dial_finish_reason(
    finish_reason: Optional[_ClaudeFinishReason],
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
