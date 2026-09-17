from collections.abc import AsyncIterator, Callable, Hashable
from typing import Literal, Protocol, TypeVar

from anthropic.types import (
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
from pydantic import BaseModel

from aidial_adapter_bedrock.anthropic_translator.errors import (
    INTERNAL_ERROR_MESSAGE,
    AnthropicErrorType,
    ErrorDetail,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class ClosableAsyncStream(Protocol[T_co]):
    def __aiter__(self) -> AsyncIterator[T_co]: ...

    async def close(self) -> None: ...


class PingEvent(BaseModel):
    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorDetail


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
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n".encode()


class AnthropicStreamState:
    def __init__(self, requested_model: str, default_message_id: str) -> None:
        self.requested_model: str = requested_model
        self.model: str = requested_model
        self.message_id: str = default_message_id
        self.next_index: int = 0
        self.terminated: bool = False

        self._open: tuple[Hashable, int] | None = None

    def is_open(self, key: Hashable) -> bool:
        return self._open is not None and self._open[0] == key

    def open_block(
        self, key: Hashable, content_block: AnthropicContentBlock
    ) -> list[bytes]:
        events: list[bytes] = self.close_block()
        index: int = self.next_index
        self.next_index += 1
        self._open = (key, index)
        events.append(
            format_sse(
                RawContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=content_block,
                )
            )
        )
        return events

    def delta(self, delta: RawContentBlockDelta) -> list[bytes]:
        if self._open is None:
            return []
        return [
            format_sse(
                RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=self._open[1],
                    delta=delta,
                )
            )
        ]

    def close_block(self) -> list[bytes]:
        if self._open is None:
            return []
        _, index = self._open
        self._open = None
        return [
            format_sse(
                RawContentBlockStopEvent(type="content_block_stop", index=index)
            )
        ]

    def message_start_events(self) -> list[bytes]:
        message: Message = Message(
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

    def final_events(
        self,
        stop_reason: StopReason,
        usage: Usage,
        stop_sequence: str | None = None,
    ) -> list[bytes]:
        if self.terminated:
            return []
        self.terminated = True
        events: list[bytes] = self.close_block()
        events.append(
            format_sse(
                RawMessageDeltaEvent(
                    type="message_delta",
                    delta=Delta(
                        stop_reason=stop_reason, stop_sequence=stop_sequence
                    ),
                    usage=MessageDeltaUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_creation_input_tokens=usage.cache_creation_input_tokens,
                        cache_read_input_tokens=usage.cache_read_input_tokens,
                        output_tokens_details=usage.output_tokens_details,
                    ),
                )
            )
        )
        events.append(format_sse(RawMessageStopEvent(type="message_stop")))
        return events

    def emit_error(
        self, error_type: AnthropicErrorType, message: str
    ) -> list[bytes]:
        if self.terminated:
            return []
        self.terminated = True
        return [
            format_sse(
                ErrorEvent(error={"type": error_type.code, "message": message})
            )
        ]


async def run_sse_stream(
    stream: ClosableAsyncStream[T],
    state: AnthropicStreamState,
    on_item: Callable[[T], list[bytes]],
    on_finalize: Callable[[], list[bytes]] | None = None,
    *,
    log_context: str,
) -> AsyncIterator[bytes]:
    try:
        async for item in stream:
            for chunk in on_item(item):
                yield chunk
        if on_finalize is not None:
            for chunk in on_finalize():
                yield chunk
    except Exception:
        log.exception(f"Error while translating the {log_context} SSE stream")
        for chunk in state.emit_error(
            AnthropicErrorType.API, INTERNAL_ERROR_MESSAGE
        ):
            yield chunk
    finally:
        await stream.close()
