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


def _text() -> TextBlock:
    return TextBlock(type="text", text="")


def _usage() -> Usage:
    return Usage(
        input_tokens=1,
        output_tokens=2,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )


def test_open_delta_close_round_trip():
    state = AnthropicStreamState("m", "id_unknown")
    [start] = state.open_block("k", _text())
    assert b'"index":0' in start
    assert state.is_open("k")
    [delta] = state.delta(TextDelta(type="text_delta", text="hi"))
    assert b'"text":"hi"' in delta
    [stop] = state.close_block()
    assert b'"content_block_stop"' in stop
    assert not state.is_open("k")


def test_delta_and_close_are_noop_when_nothing_is_open():
    state = AnthropicStreamState("m", "id_unknown")
    assert state.delta(TextDelta(type="text_delta", text="")) == []
    assert state.close_block() == []


def test_opening_a_block_closes_the_previous_one():
    # Anthropic allows only one open content block at a time.
    state = AnthropicStreamState("m", "id_unknown")
    state.open_block("a", _text())
    events = state.open_block("b", _text())
    assert len(events) == 2
    assert b'"content_block_stop"' in events[0]
    assert b'"index":0' in events[0]
    assert b'"content_block_start"' in events[1]
    assert b'"index":1' in events[1]
    assert state.is_open("b")
    assert not state.is_open("a")


def test_the_block_index_is_a_single_flat_counter():
    state = AnthropicStreamState("m", "id_unknown")
    for expected in range(3):
        # Past the first, `open_block` also closes the previous block.
        start = state.open_block(expected, _text())[-1]
        assert f'"index":{expected}'.encode() in start


def test_message_start_events_use_message_id_and_model():
    state = AnthropicStreamState("requested-model", "fallback_id")
    state.model = "resolved-model"
    state.message_id = "resolved_id"
    start, ping = state.message_start_events()
    assert b"resolved_id" in start
    assert b"resolved-model" in start
    assert b'"ping"' in ping


def test_final_events_close_the_open_block_and_terminate():
    state = AnthropicStreamState("m", "id_unknown")
    state.open_block("k", _text())
    events = state.final_events("end_turn", None, _usage())
    assert len(events) == 3  # content_block_stop, message_delta, message_stop
    assert b"content_block_stop" in events[0]
    assert b'"stop_reason":"end_turn"' in events[1]
    assert b"message_stop" in events[2]
    assert not state.is_open("k")


def test_final_events_carry_the_matched_stop_sequence():
    state = AnthropicStreamState("m", "id_unknown")
    events = state.final_events("stop_sequence", "STOP", _usage())
    assert b'"stop_sequence":"STOP"' in events[0]


def test_an_error_after_termination_is_suppressed():
    # Nothing may follow message_stop, including a fault raised by an upstream
    # that keeps failing past its own finish_reason.
    state = AnthropicStreamState("m", "id_unknown")
    assert state.emit_error("api_error", "boom") != []
    state.final_events("end_turn", None, _usage())
    assert state.emit_error("api_error", "boom") == []


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
    # Once streaming has started there is no HTTP status left to report with.
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
