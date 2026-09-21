import uuid
from collections.abc import Mapping
from typing import assert_never

from anthropic.types import (
    Message,
    ServerToolUseBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
    WebSearchToolResultBlock,
)
from anthropic.types.content_block import ContentBlock as AnthropicContentBlock
from anthropic.types.output_tokens_details import OutputTokensDetails
from anthropic.types.web_search_result_block import WebSearchResultBlock
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import (
    Annotation,
    AnnotationURLCitation,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from pydantic import TypeAdapter, ValidationError

from aidial_adapter_bedrock.anthropic_translator.chat_completions.dial_extensions import (
    CustomContent,
    DialExtras,
    parse_extras,
    signed_thinking,
    stage_thinking,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_ARGUMENTS: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])

UNKNOWN_MESSAGE_ID: str = "chatcmpl_unknown"


def from_chat_completions(
    response: ChatCompletion,
    requested_model: str,
    aliases: ToolNameAliases,
) -> Message:
    choice: Choice | None = response.choices[0] if response.choices else None
    message: ChatCompletionMessage | None = choice.message if choice else None
    tool_blocks: list[ToolUseBlock] = _tool_use_blocks(
        message.tool_calls if message else None, aliases
    )
    refusal: str | None = message.refusal if message else None

    content: list[AnthropicContentBlock] = _content_blocks(message, tool_blocks)
    return Message(
        id=getattr(response, "id", None) or UNKNOWN_MESSAGE_ID,
        type="message",
        role="assistant",
        model=getattr(response, "model", None) or requested_model,
        content=content,
        stop_sequence=None,
        stop_reason=stop_reason(
            choice.finish_reason if choice else None,
            bool(tool_blocks),
            bool(refusal),
        ),
        usage=convert_usage(response.usage),
    )


def _content_blocks(
    message: ChatCompletionMessage | None,
    tool_blocks: list[ToolUseBlock],
) -> list[AnthropicContentBlock]:
    if message is None:
        return [_empty_text()]

    content: list[AnthropicContentBlock] = []

    extras: DialExtras = parse_extras(message.model_extra)
    if thinking := thinking_block(extras.custom_content):
        content.append(thinking)

    content.extend(citation_blocks(_citations(message.annotations)))

    if message.content:
        content.append(TextBlock(type="text", text=message.content))

    if message.refusal:
        content.append(TextBlock(type="text", text=message.refusal))

    content.extend(tool_blocks)

    return content or [_empty_text()]


def _empty_text() -> TextBlock:
    return TextBlock(type="text", text="")


def thinking_block(
    custom_content: CustomContent | None,
) -> ThinkingBlock | None:
    if block := signed_thinking(custom_content):
        return ThinkingBlock(
            type="thinking",
            thinking=block.thinking or "",
            signature=block.signature or "",
        )
    if text := stage_thinking(custom_content):
        return ThinkingBlock(type="thinking", thinking=text, signature="")
    return None


def _citations(annotations: list[Annotation] | None) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for annotation in annotations or []:
        citation: AnnotationURLCitation = annotation.url_citation
        if annotation.type != "url_citation" or not citation.url:
            log.warning("Skipping malformed annotation: %s", annotation.type)
            continue
        result.append((citation.url, citation.title or ""))
    return result


def citation_blocks(
    citations: list[tuple[str, str]],
) -> list[AnthropicContentBlock]:
    if not citations:
        return []
    tool_use_id: str = f"srvtoolu_{uuid.uuid4().hex}"
    return [
        ServerToolUseBlock(
            type="server_tool_use",
            id=tool_use_id,
            name="web_search",
            input={"query": ""},
        ),
        WebSearchToolResultBlock(
            type="web_search_tool_result",
            tool_use_id=tool_use_id,
            content=[
                WebSearchResultBlock(
                    type="web_search_result",
                    url=url,
                    title=title,
                    encrypted_content="",
                )
                for url, title in citations
            ],
        ),
    ]


def _tool_use_blocks(
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None,
    aliases: ToolNameAliases,
) -> list[ToolUseBlock]:
    blocks: list[ToolUseBlock] = []
    for call in tool_calls or []:
        match call.type:
            case "custom":
                log.warning(
                    "Skipping unsupported tool call type: %s", call.type
                )
                continue
            case "function":
                pass
            case unexpected:
                assert_never(unexpected)
        if not call.id or not call.function.name:
            log.warning("Skipping tool call without an id or name")
            continue
        blocks.append(
            ToolUseBlock(
                type="tool_use",
                id=call.id,
                name=aliases.to_client(call.function.name),
                input=parse_arguments(call.function.arguments),
            )
        )
    return blocks


def parse_arguments(arguments: str | None) -> dict[str, object]:
    if not arguments:
        return {}
    try:
        return _ARGUMENTS.validate_json(arguments)
    except ValidationError:
        log.warning("Failed to parse function_call arguments as JSON")
        return {}


def stop_reason(
    finish_reason: str | None,
    saw_tool_use: bool,
    saw_refusal: bool,
) -> StopReason:
    if saw_tool_use:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter" or saw_refusal:
        return "refusal"
    return "end_turn"


_CACHE_WRITE_KEYS: tuple[str, ...] = ("cache_write_tokens", "cacheWriteTokens")


def convert_usage(usage: CompletionUsage | None) -> Usage:
    prompt_details: PromptTokensDetails | None = (
        usage.prompt_tokens_details if usage else None
    )
    cache_read: int = (
        prompt_details.cached_tokens if prompt_details else 0
    ) or 0
    cache_write: int = _cache_write_tokens(prompt_details)
    prompt_tokens: int = getattr(usage, "prompt_tokens", 0) or 0

    completion_details: CompletionTokensDetails | None = (
        usage.completion_tokens_details if usage else None
    )
    thinking: int = (
        completion_details.reasoning_tokens if completion_details else 0
    ) or 0

    return Usage(
        input_tokens=max(prompt_tokens - cache_read - cache_write, 0),
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        output_tokens_details=(
            OutputTokensDetails(thinking_tokens=thinking) if thinking else None
        ),
    )


def _cache_write_tokens(details: PromptTokensDetails | None) -> int:
    if details is not None and details.cache_write_tokens is not None:
        return details.cache_write_tokens

    extra: Mapping[str, object] = (
        details.model_extra if details else None
    ) or {}
    for key in _CACHE_WRITE_KEYS:
        if isinstance(value := extra.get(key), int):
            return value
    return 0
