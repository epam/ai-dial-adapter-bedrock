"""
Non-streaming OpenAI Chat Completions object → Anthropic Message.

Also home to the pieces the streaming translator must share verbatim — stop
reason, usage arithmetic and citation blocks — because a divergence between the
two modes is a bug that manifests in only one of them and is easily missed.
"""

import json
import uuid
from typing import Any

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
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from aidial_adapter_bedrock.anthropic_translator.chat_completions.dial_extensions import (
    CustomContent,
    parse_extras,
    signed_thinking,
    stage_thinking,
)
from aidial_adapter_bedrock.anthropic_translator.stop_sequences import (
    StopMatch,
    apply_stop_sequences,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

# Fallback message id when the upstream chunk/response carries none.
UNKNOWN_MESSAGE_ID = "chatcmpl_unknown"


def from_chat_completions(
    response: ChatCompletion,
    requested_model: str,
    aliases: ToolNameAliases,
    stop_sequences: list[str],
) -> Message:
    """Only the first choice is translated: Anthropic has no `n`-choices
    concept, and this translator never requests more than one."""
    tlog = TranslationLog("Chat Completions→Anthropic response")
    try:
        choice: Choice | None = (
            response.choices[0] if response.choices else None
        )
        message: ChatCompletionMessage | None = (
            choice.message if choice else None
        )
        stop: StopMatch = apply_stop_sequences(
            message.content or "" if message else "", stop_sequences
        )
        tool_blocks: list[ToolUseBlock] = _tool_use_blocks(
            message.tool_calls if message else None, aliases, tlog
        )
        refusal: str | None = message.refusal if message else None

        return Message(
            id=response.id or UNKNOWN_MESSAGE_ID,
            type="message",
            role="assistant",
            model=response.model or requested_model,
            content=_content_blocks(message, stop.text, tool_blocks, tlog),
            stop_reason=stop_reason(
                choice.finish_reason if choice else None,
                stop.sequence,
                bool(tool_blocks),
                bool(refusal),
            ),
            stop_sequence=stop.sequence,
            usage=convert_usage(response.usage),
        )
    finally:
        tlog.flush()


def _content_blocks(
    message: ChatCompletionMessage | None,
    text: str,
    tool_blocks: list[ToolUseBlock],
    tlog: TranslationLog,
) -> list[AnthropicContentBlock]:
    """The blocks in the order Anthropic fixes — thinking, web-search pair,
    text, refusal, tool calls. Thinking must lead: a misordered signed thinking
    block is rejected when the turn is replayed.
    """
    if message is None:
        return [_empty_text()]

    content: list[AnthropicContentBlock] = []

    extras = parse_extras(message.model_extra)
    if thinking := thinking_block(extras.custom_content):
        content.append(thinking)

    for url, title in _citations(message.annotations, tlog):
        content.extend(citation_blocks(url, title))

    if text:
        content.append(TextBlock(type="text", text=text))

    if message.refusal:
        content.append(TextBlock(type="text", text=message.refusal))

    content.extend(tool_blocks)

    # Anthropic messages always carry at least one block and SDKs index
    # `content[0]` unconditionally.
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
        # A signature is required; an empty one round-trips and cannot be
        # mistaken for a real one.
        return ThinkingBlock(type="thinking", thinking=text, signature="")
    return None


def _citations(
    annotations: list[Annotation] | None, tlog: TranslationLog
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for annotation in annotations or []:
        citation = annotation.url_citation
        if annotation.type != "url_citation" or not citation.url:
            tlog.warning("Skipping malformed annotation: %s", annotation.type)
            continue
        result.append((citation.url, citation.title or ""))
    return result


def citation_blocks(url: str, title: str) -> list[AnthropicContentBlock]:
    """A web-search call and its result, synthesised from an annotation that
    recorded only the result.

    `input.query` is empty because the annotation never carries the query, and
    `encrypted_content` because it has no upstream source; both are required by
    the schema.
    """
    tool_use_id = f"srvtoolu_{uuid.uuid4().hex}"
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
            ],
        ),
    ]


def _tool_use_blocks(
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None,
    aliases: ToolNameAliases,
    tlog: TranslationLog,
) -> list[ToolUseBlock]:
    blocks: list[ToolUseBlock] = []
    for call in tool_calls or []:
        if call.type != "function":
            tlog.warning("Skipping unsupported tool call type: %s", call.type)
            continue
        if not call.id or not call.function.name:
            # The client could never correlate a result with a half-built one.
            tlog.warning("Skipping tool call without an id or name")
            continue
        blocks.append(
            ToolUseBlock(
                type="tool_use",
                id=call.id,
                name=aliases.to_client(call.function.name),
                input=parse_arguments(call.function.arguments, tlog),
            )
        )
    return blocks


def parse_arguments(
    arguments: str | None, tlog: TranslationLog
) -> dict[str, Any]:
    """A malformed tool call must not take down the whole response."""
    if not arguments:
        return {}
    try:
        parsed: Any = json.loads(arguments)
    except json.JSONDecodeError:
        tlog.warning("Failed to parse function_call arguments as JSON")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def stop_reason(
    finish_reason: str | None,
    matched_stop: str | None,
    saw_tool_use: bool,
    saw_refusal: bool,
) -> StopReason:
    """Map a Chat Completions `finish_reason`, plus what the content revealed,
    to an Anthropic `stop_reason`. Never `null`: clients choke on it."""
    if matched_stop is not None:
        return "stop_sequence"
    if saw_tool_use:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter" or saw_refusal:
        return "refusal"
    return "end_turn"


# Neither spelling is part of OpenAI's schema.
_CACHE_WRITE_KEYS = ("cache_write_tokens", "cacheWriteTokens")


def convert_usage(usage: CompletionUsage | None) -> Usage:
    """Anthropic reports cache reads *and* cache writes *outside*
    `input_tokens` while OpenAI counts both *inside* `prompt_tokens`, so the
    subtraction is mandatory: forwarding `prompt_tokens` verbatim double-counts
    every cached token and inflates the cost the client displays.

    Every counter reads as 0 when absent or null — Core serialises the empty
    ones explicitly — so missing usage is never an error.
    """
    prompt_details = usage.prompt_tokens_details if usage else None
    cache_read = (prompt_details.cached_tokens if prompt_details else 0) or 0
    cache_write = _cache_write_tokens(prompt_details)
    prompt_tokens = usage.prompt_tokens if usage else 0

    completion_details = usage.completion_tokens_details if usage else None
    thinking = (
        completion_details.reasoning_tokens if completion_details else 0
    ) or 0

    return Usage(
        input_tokens=max(prompt_tokens - cache_read - cache_write, 0),
        output_tokens=usage.completion_tokens if usage else 0,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
        # Already inside `completion_tokens`, so this only decomposes it.
        output_tokens_details=(
            OutputTokensDetails(thinking_tokens=thinking) if thinking else None
        ),
    )


def _cache_write_tokens(details: PromptTokensDetails | None) -> int:
    extra = (details.model_extra if details else None) or {}
    for key in _CACHE_WRITE_KEYS:
        if isinstance(value := extra.get(key), int):
            return value
    return 0
