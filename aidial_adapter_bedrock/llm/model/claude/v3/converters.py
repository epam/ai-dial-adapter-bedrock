from typing import Iterable, List, Literal, Optional, Set, Tuple, assert_never

from aidial_sdk.chat_completion import FinishReason, Tool
from aidial_sdk.chat_completion import ToolChoice as DialToolChoice
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControlEphemeralParam,
)
from anthropic.types.beta import BetaContentBlockParam as ContentBlockParam
from anthropic.types.beta import BetaImageBlockParam as ImageBlockParam
from anthropic.types.beta import BetaMessageParam as MessageParam
from anthropic.types.beta import (
    BetaRequestDocumentBlockParam as RequestDocumentBlockParam,
)
from anthropic.types.beta import BetaStopReason as ClaudeStopReason
from anthropic.types.beta import BetaTextBlockParam as TextBlockParam
from anthropic.types.beta import BetaToolChoiceAnyParam as ToolChoiceAnyParam
from anthropic.types.beta import BetaToolChoiceAutoParam as ToolChoiceAutoParam
from anthropic.types.beta import BetaToolChoiceNoneParam as ToolChoiceNoneParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolChoiceToolParam as ToolChoiceToolParam
from anthropic.types.beta import BetaToolParam as ToolParam
from anthropic.types.beta import BetaUsage as Usage
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
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
    AttachmentProcessors,
)
from aidial_adapter_bedrock.llm.model.claude.v3.blocks import (
    create_text_block,
    create_tool_result_block,
    create_tool_use_block,
)
from aidial_adapter_bedrock.llm.model.claude.v3.config import Configuration
from aidial_adapter_bedrock.llm.model.claude.v3.state import (
    get_message_content_from_state,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig, ToolsMode
from aidial_adapter_bedrock.utils.list import group_by
from aidial_adapter_bedrock.utils.list_projection import ListProjection

_claude_cache_breakpoint = CacheControlEphemeralParam(type="ephemeral")


def _add_cache_control(
    message: BaseMessage | HumanToolResultMessage | AIToolCallMessage,
    claude_content: Iterable[ContentBlockParam],
) -> Iterable[ContentBlockParam]:

    if message.cache_breakpoint is not None:
        for block in reversed(list(claude_content)):
            if (
                isinstance(block, dict)
                and block["type"] != "thinking"
                and block["type"] != "redacted_thinking"
            ):
                block["cache_control"] = _claude_cache_breakpoint
                break

    return claude_content


def _get_claude_message_role(
    dial_message: (
        AIRegularMessage
        | AIToolCallMessage
        | HumanRegularMessage
        | HumanToolResultMessage
    ),
) -> Literal["assistant", "user"]:
    match dial_message:
        case AIRegularMessage() | AIToolCallMessage():
            return "assistant"
        case HumanRegularMessage() | HumanToolResultMessage():
            return "user"
        case _:
            assert_never(dial_message)


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
    handlers: AttachmentProcessors[
        TextBlockParam | ImageBlockParam | RequestDocumentBlockParam,
        Configuration,
    ],
    messages: List[BaseMessage | HumanToolResultMessage | AIToolCallMessage],
) -> Tuple[List[TextBlockParam], ListProjection[MessageParam]]:

    idx_offset: int = 0
    system_messages: List[TextBlockParam] = []

    for message in messages:
        if not isinstance(message, SystemMessage):
            break

        idx_offset += 1
        content = await handlers.process_attachments(message)
        content = _add_cache_control(message, content)
        system_messages.extend(content)  # type: ignore

    claude_messages: ListProjection[MessageParam] = ListProjection()

    for idx, message in enumerate(messages[idx_offset:], start=idx_offset):

        match message:
            case HumanRegularMessage():
                content = await handlers.process_attachments(message)

            case AIRegularMessage():
                # Take the message content from the state if possible,
                # since it may include certain content blocks that
                # are missing from the DIAL message itself,
                # such as thinking signatures and redacted thinking blocks.
                content = get_message_content_from_state(idx, message)
                if content is None:
                    content = await handlers.process_attachments(message)

            case AIToolCallMessage():
                content = get_message_content_from_state(idx, message)

                if content is None:
                    content = [
                        create_tool_use_block(call) for call in message.calls
                    ]
                    if text_content := message.content:
                        content.insert(0, create_text_block(text_content))

            case HumanToolResultMessage():
                content = [create_tool_result_block(message)]

            case SystemMessage():
                raise ValidationError(
                    "System and developer messages are only allowed in the begging of the conversation."
                )
            case _:
                assert_never(message)

        claude_message = MessageParam(
            role=_get_claude_message_role(message),
            content=_add_cache_control(message, content),
        )

        claude_messages.append(claude_message, idx)

    return system_messages, _merge_messages_with_same_role(claude_messages)


def to_dial_finish_reason(
    finish_reason: Optional[ClaudeStopReason],
    tools_mode: ToolsMode | None,
) -> FinishReason:
    if finish_reason is None:
        return FinishReason.STOP

    match finish_reason:
        case "end_turn":
            return FinishReason.STOP
        case "max_tokens" | "model_context_window_exceeded":
            return FinishReason.LENGTH
        case "stop_sequence" | "pause_turn" | "refusal":
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


def _to_claude_tool(tool: Tool) -> ToolParam:
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


def _to_claude_tool_choice(
    tool_choice: Literal["auto", "none", "required"] | DialToolChoice,
) -> ToolChoice:
    # NOTE tool_choice.disable_parallel_tool_use=True option isn't supported
    # by older Claude3 versions, so we limit the number of generated function calls
    # to one in the adapter itself for the functions mode.

    match tool_choice:
        case DialToolChoice(function=function):
            return ToolChoiceToolParam(type="tool", name=function.name)
        case "required":
            return ToolChoiceAnyParam(type="any")
        case "auto":
            return ToolChoiceAutoParam(type="auto")
        case "none":
            return ToolChoiceNoneParam(type="none")
        case _:
            assert_never(tool_choice)


class ClaudeToolsConfig(BaseModel):
    tools: List[ToolParam]
    tool_choice: ToolChoice


def to_claude_tool_config(
    tools_config: ToolsConfig | None,
) -> ClaudeToolsConfig | None:
    if tools_config is None or not tools_config.tools:
        return None

    tools = [_to_claude_tool(tool) for tool in tools_config.tools]
    tool_choice = _to_claude_tool_choice(tools_config.tool_choice)
    return ClaudeToolsConfig(tools=tools, tool_choice=tool_choice)
