"""
Non-streaming OpenAI Chat Completions object → Anthropic Message.
"""

import json
from typing import Any

from anthropic.types import Message, StopReason, TextBlock, ToolUseBlock, Usage
from anthropic.types.content_block import ContentBlock as AnthropicContentBlock
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

# Fallback message id when the upstream chunk/response carries none.
UNKNOWN_MESSAGE_ID = "chatcmpl_unknown"


def from_chat_completions(
    response: ChatCompletion, requested_model: str
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

        content, saw_refusal = _convert_message(message, tlog)
        reason: StopReason = stop_reason(
            choice.finish_reason if choice else None, saw_refusal
        )

        return Message(
            id=response.id or UNKNOWN_MESSAGE_ID,
            type="message",
            role="assistant",
            model=response.model or requested_model,
            content=content,
            stop_reason=reason,
            stop_sequence=None,
            usage=convert_usage(response.usage),
        )
    finally:
        tlog.flush()


def _convert_message(
    message: ChatCompletionMessage | None, tlog: TranslationLog
) -> tuple[list[AnthropicContentBlock], bool]:
    content: list[AnthropicContentBlock] = []
    saw_refusal = False

    if message is None:
        return content, saw_refusal

    if text := message.content:
        content.append(TextBlock(type="text", text=text))

    if refusal := message.refusal:
        saw_refusal = True
        content.append(TextBlock(type="text", text=refusal))

    for call in message.tool_calls or []:
        if call.type != "function":
            raise ValueError(f"Unsupported tool call type: {call.type!r}")
        if not call.id or not call.function.name:
            raise ValueError("Tool call is missing its required id or name")
        content.append(
            ToolUseBlock(
                type="tool_use",
                id=call.id,
                name=call.function.name,
                input=_parse_arguments(call.function.arguments, tlog),
            )
        )

    return content, saw_refusal


def _parse_arguments(
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


def stop_reason(finish_reason: str | None, saw_refusal: bool) -> StopReason:
    """Map a Chat Completions `finish_reason` (plus what the content revealed) to an Anthropic `stop_reason`."""
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "content_filter" or saw_refusal:
        return "refusal"
    # Unrecognized/missing finish reasons land here: never emit null, Claude Code chokes on it.
    return "end_turn"


def convert_usage(usage: CompletionUsage | None) -> Usage:
    details = usage.prompt_tokens_details if usage else None
    return Usage(
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        cache_read_input_tokens=(details.cached_tokens if details else 0) or 0,
        cache_creation_input_tokens=0,
    )
