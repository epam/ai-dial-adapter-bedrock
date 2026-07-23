"""
SSE state-machine plumbing for Anthropic-shaped streaming responses.

A translator turns its differently-shaped upstream event/chunk stream into the
same Anthropic SSE event sequence:

    message_start -> ping
        -> (content_block_start -> delta* -> content_block_stop)*
        -> message_delta -> message_stop

This module owns everything about that sequence that doesn't depend on the
upstream shape: SSE formatting, the monotonically increasing Anthropic
content-block index (keyed by an arbitrary per-translator coordinate),
`message_start`/`ping` construction, error emission, and the final
`message_delta`/`message_stop` assembly. Each translator subclasses
`AnthropicStreamState` and supplies only its own event dispatch and its own
stop-reason/usage computation.

Every event this module emits is a real `anthropic.types` pydantic model
except `PingEvent` and `ErrorEvent`, which have no SDK equivalent (see their
docstrings for why).
"""

from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, TypeVar

import openai
from anthropic.types import (
    APIErrorObject,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    StopReason,
    Usage,
)
from anthropic.types.message_delta_usage import MessageDeltaUsage
from anthropic.types.raw_content_block_delta import RawContentBlockDelta
from anthropic.types.raw_content_block_start_event import (
    ContentBlock as AnthropicContentBlock,
)
from anthropic.types.raw_message_delta_event import Delta
from openai import AsyncStream
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

T = TypeVar("T")


class PingEvent(BaseModel):
    """Anthropic's keep-alive SSE event. Not modeled by the Anthropic SDK —
    it carries no data and exists purely at the transport level."""

    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """Anthropic's terminal streaming error event. The wrapper itself isn't
    modeled by the Anthropic SDK — a mid-stream failure has no corresponding
    response schema to validate against — but `error` is the SDK's own
    `APIErrorObject`, since every error this translator emits is that kind."""

    type: Literal["error"] = "error"
    error: APIErrorObject


AnthropicSSEEvent = (
    RawMessageStartEvent
    | RawContentBlockStartEvent
    | RawContentBlockDeltaEvent
    | RawContentBlockStopEvent
    | RawMessageDeltaEvent
    | RawMessageStopEvent
    | PingEvent
    | ErrorEvent
)


def format_sse(event: AnthropicSSEEvent) -> bytes:
    """The `event:` line is redundant with the `type` field already inside
    `data:`, matching what the Anthropic SDKs expect."""
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n".encode()


class AnthropicStreamState:
    """Base state for translating an upstream stream into Anthropic SSE."""

    def __init__(self, requested_model: str, default_message_id: str):
        self.requested_model = requested_model
        self.model = requested_model
        self.message_id = default_message_id
        self.next_index = 0
        # coordinate key -> Anthropic content-block index (only while open)
        self.block_index: dict[Any, int] = {}
        self.saw_function_call = False
        self.saw_refusal = False

    def _open_block(
        self, key: Any, content_block: AnthropicContentBlock
    ) -> bytes:
        index: int = self.next_index
        self.next_index += 1
        self.block_index[key] = index
        return format_sse(
            RawContentBlockStartEvent(
                type="content_block_start",
                index=index,
                content_block=content_block,
            )
        )

    def _delta(self, key: Any, delta: RawContentBlockDelta) -> list[bytes]:
        index: int | None = self.block_index.get(key)
        if index is None:
            return []
        return [
            format_sse(
                RawContentBlockDeltaEvent(
                    type="content_block_delta", index=index, delta=delta
                )
            )
        ]

    def _close_block(self, key: Any) -> list[bytes]:
        index: int | None = self.block_index.pop(key, None)
        if index is None:
            return []
        return [
            format_sse(
                RawContentBlockStopEvent(type="content_block_stop", index=index)
            )
        ]

    def _close_all_open(self) -> list[bytes]:
        events: list[bytes] = [
            format_sse(
                RawContentBlockStopEvent(type="content_block_stop", index=index)
            )
            for _, index in sorted(
                self.block_index.items(), key=lambda kv: kv[1]
            )
        ]
        self.block_index.clear()
        return events

    def message_start_events(self) -> list[bytes]:
        message = Message(
            id=self.message_id,
            type="message",
            role="assistant",
            model=self.model,
            content=[],
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        return [
            format_sse(
                RawMessageStartEvent(type="message_start", message=message)
            ),
            format_sse(PingEvent()),
        ]

    def _on_final(self, stop_reason: StopReason, usage: Usage) -> list[bytes]:
        events: list[bytes] = self._close_all_open()
        events.append(
            format_sse(
                RawMessageDeltaEvent(
                    type="message_delta",
                    delta=Delta(stop_reason=stop_reason, stop_sequence=None),
                    usage=MessageDeltaUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_creation_input_tokens=usage.cache_creation_input_tokens,
                        cache_read_input_tokens=usage.cache_read_input_tokens,
                    ),
                )
            )
        )
        events.append(format_sse(RawMessageStopEvent(type="message_stop")))
        return events

    def emit_error(
        self, type: Literal["api_error"], message: str
    ) -> list[bytes]:
        return [
            format_sse(
                ErrorEvent(error=APIErrorObject(type=type, message=message))
            )
        ]


async def run_sse_stream(
    stream: AsyncStream[T],
    state: AnthropicStreamState,
    on_item: Callable[[T], list[bytes]],
    on_finalize: Callable[[], list[bytes]] | None = None,
    *,
    log_context: str,
) -> AsyncIterator[bytes]:
    """Consume `stream`, dispatching each item to `on_item`, closing the
    stream in every case, and turning any exception into a terminal SSE
    `error` event via `state.emit_error` instead of letting it propagate —
    once streaming has started there is no HTTP status left to report it
    with.

    `on_finalize`, when given, runs once after the loop ends normally (not on
    an exception) to emit any terminal events the stream itself didn't.
    """
    try:
        async for item in stream:
            for chunk in on_item(item):
                yield chunk
        if on_finalize is not None:
            for chunk in on_finalize():
                yield chunk
    except Exception as e:  # noqa: BLE001 — must surface as an SSE error
        log.exception(f"Error while translating the {log_context} SSE stream")
        message: str = e.message if isinstance(e, openai.APIError) else str(e)
        for chunk in state.emit_error("api_error", message):
            yield chunk
    finally:
        await stream.close()
