from typing import cast

from anthropic.types import TextBlock, Usage
from anthropic.types.text_delta import TextDelta
from openai import AsyncStream

from aidial_adapter_bedrock.anthropic_translator.sse_stream import (
    AnthropicStreamState,
    run_sse_stream,
)


class _FakeStream:
    def __init__(self, items, raise_after: Exception | None = None):
        self._items = list(items)
        self._raise_after = raise_after
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._items:
            yield item
        if self._raise_after is not None:
            raise self._raise_after

    async def close(self):
        self.closed = True


def test_open_delta_close_round_trip():
    state = AnthropicStreamState("m", "id_unknown")
    start = state._open_block("k", TextBlock(type="text", text=""))
    assert b'"index":0' in start
    [delta] = state._delta("k", TextDelta(type="text_delta", text="hi"))
    assert b'"text":"hi"' in delta
    [stop] = state._close_block("k")
    assert b'"content_block_stop"' in stop
    assert state.block_index == {}


def test_delta_and_close_are_noop_for_unknown_key():
    state = AnthropicStreamState("m", "id_unknown")
    assert state._delta("missing", TextDelta(type="text_delta", text="")) == []
    assert state._close_block("missing") == []


def test_close_all_open_closes_in_index_order():
    state = AnthropicStreamState("m", "id_unknown")
    state._open_block("b", TextBlock(type="text", text=""))
    state._open_block("a", TextBlock(type="text", text=""))
    events = state._close_all_open()
    assert len(events) == 2
    assert b'"index":0' in events[0]
    assert b'"index":1' in events[1]
    assert state.block_index == {}


def test_message_start_events_use_message_id_and_model():
    state = AnthropicStreamState("requested-model", "fallback_id")
    state.model = "resolved-model"
    state.message_id = "resolved_id"
    start, ping = state.message_start_events()
    assert b"resolved_id" in start
    assert b"resolved-model" in start
    assert b'"ping"' in ping


def test_on_final_closes_open_blocks_and_emits_delta_and_stop():
    state = AnthropicStreamState("m", "id_unknown")
    state._open_block("k", TextBlock(type="text", text=""))
    events = state._on_final(
        "end_turn",
        Usage(
            input_tokens=1,
            output_tokens=2,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    assert len(events) == 3  # content_block_stop, message_delta, message_stop
    assert b"content_block_stop" in events[0]
    assert b'"stop_reason":"end_turn"' in events[1]
    assert b"message_stop" in events[2]
    assert state.block_index == {}


async def test_run_sse_stream_dispatches_items_and_closes_stream():
    stream = _FakeStream(["a", "b"])
    state = AnthropicStreamState("m", "id_unknown")
    chunks = [
        chunk
        async for chunk in run_sse_stream(
            cast(AsyncStream[str], stream),
            state,
            lambda item: [item.encode()],
            log_context="Test",
        )
    ]
    assert chunks == [b"a", b"b"]
    assert stream.closed is True


async def test_run_sse_stream_calls_finalize_after_normal_completion():
    stream = _FakeStream(["a"])
    state = AnthropicStreamState("m", "id_unknown")
    chunks = [
        chunk
        async for chunk in run_sse_stream(
            cast(AsyncStream[str], stream),
            state,
            lambda item: [item.encode()],
            on_finalize=lambda: [b"done"],
            log_context="Test",
        )
    ]
    assert chunks == [b"a", b"done"]


async def test_run_sse_stream_turns_exception_into_error_event():
    stream = _FakeStream(["a"], raise_after=RuntimeError("boom"))
    state = AnthropicStreamState("m", "id_unknown")
    chunks = [
        chunk
        async for chunk in run_sse_stream(
            cast(AsyncStream[str], stream),
            state,
            lambda item: [item.encode()],
            log_context="Test",
        )
    ]
    assert chunks[0] == b"a"
    assert b'"error"' in chunks[1]
    assert b"boom" in chunks[1]
    assert stream.closed is True
