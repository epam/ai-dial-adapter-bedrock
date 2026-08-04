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
from openai.types.completion_usage import CompletionUsage

from aidial_adapter_bedrock.anthropic_translator.chat_completions.dial_extensions import (
    CustomContent,
    parse_extras,
    signed_thinking,
    stage_thinking,
)
from aidial_adapter_bedrock.anthropic_translator.stop_sequences import (
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

        content, matched_stop, saw_refusal, saw_tool_use = _convert_message(
            message, aliases, stop_sequences, tlog
        )

        return Message(
            id=response.id or UNKNOWN_MESSAGE_ID,
            type="message",
            role="assistant",
            model=response.model or requested_model,
            content=content,
            stop_reason=stop_reason(
                choice.finish_reason if choice else None,
                matched_stop,
                saw_tool_use,
                saw_refusal,
            ),
            stop_sequence=matched_stop,
            usage=convert_usage(response.usage),
        )
    finally:
        tlog.flush()


def _convert_message(
    message: ChatCompletionMessage | None,
    aliases: ToolNameAliases,
    stop_sequences: list[str],
    tlog: TranslationLog,
) -> tuple[list[AnthropicContentBlock], str | None, bool, bool]:
    """The content blocks in the order Anthropic fixes — thinking, web-search
    pair, text, refusal, tool calls — with what they revealed about the stop
    reason. Thinking must lead: a misordered signed thinking block is rejected
    when the turn is replayed.
    """
    content: list[AnthropicContentBlock] = []
    matched_stop: str | None = None
    saw_refusal = False

    if message is None:
        return [_empty_text()], None, False, False

    extras = parse_extras(message.model_extra)
    if thinking := thinking_block(extras.custom_content):
        content.append(thinking)

    for url, title in _citations(message.annotations, tlog):
        content.extend(citation_blocks(url, title))

    if text := message.content:
        text, matched_stop = apply_stop_sequences(text, stop_sequences)
        if text:
            content.append(TextBlock(type="text", text=text))

    if refusal := message.refusal:
        saw_refusal = True
        content.append(TextBlock(type="text", text=refusal))

    tool_blocks = _tool_use_blocks(message.tool_calls, aliases, tlog)
    content.extend(tool_blocks)

    # Anthropic messages always carry at least one block, and SDKs index
    # `content[0]` unconditionally, so an empty completion — or one whose
    # budget reasoning consumed entirely — is backfilled.
    return (
        content or [_empty_text()],
        matched_stop,
        saw_refusal,
        bool(tool_blocks),
    )


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
        # Anthropic requires a signature on the block. An empty one round-trips
        # and cannot be mistaken for a real one.
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
            # A half-built block the client could never correlate a result
            # with is worse than none.
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


def convert_usage(usage: CompletionUsage | None) -> Usage:
    """Anthropic reports cache reads *outside* `input_tokens` while OpenAI
    counts them *inside* `prompt_tokens`, so the subtraction is mandatory:
    forwarding `prompt_tokens` verbatim double-counts every cached token and
    inflates the cost the client displays.

    `cache_creation_input_tokens` stays 0 because cache writes are folded into
    `prompt_tokens` with nothing to distinguish them, and a guess is worse.
    """
    details = usage.prompt_tokens_details if usage else None
    cache_read = (details.cached_tokens if details else 0) or 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    return Usage(
        input_tokens=max(prompt_tokens - cache_read, 0),
        output_tokens=usage.completion_tokens if usage else 0,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=0,
    )
