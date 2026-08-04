import json
from typing import Any, Literal, cast

import pytest
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

from aidial_adapter_bedrock.anthropic_translator.chat_completions.streaming import (
    translate_stream,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

LONG_MCP_NAME = "mcp__" + "s" * 60 + "__do_the_thing"

Event = tuple[str, dict]


class FakeStream:
    """A minimal stand-in for `openai.AsyncStream`: async-iterable over
    pre-built `ChatCompletionChunk`s, with a no-op `close()`."""

    def __init__(self, items: list[ChatCompletionChunk]):
        self._items = list(items)
        self.closed = False

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._items:
            yield item

    async def close(self):
        self.closed = True


def parse_anthropic_sse(raw: bytes) -> list[Event]:
    out: list[Event] = []
    for block in raw.decode().split("\n\n"):
        if not block.strip():
            continue
        event = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        assert event is not None
        out.append((event, json.loads("\n".join(data_lines))))
    return out


async def translate(
    chunks: list[ChatCompletionChunk],
    model: str = "gpt-5.5",
    aliases: ToolNameAliases | None = None,
    stop_sequences: list[str] | None = None,
) -> list[Event]:
    stream = FakeStream(chunks)
    collected = b""
    async for chunk in translate_stream(
        cast(AsyncStream[ChatCompletionChunk], stream),
        model,
        aliases or ToolNameAliases(),
        stop_sequences or [],
    ):
        collected += chunk
    return parse_anthropic_sse(collected)


def chunk(
    delta: ChoiceDelta | dict[str, Any] | None = None,
    finish_reason: FinishReason | None = None,
    id: str = "chatcmpl_1",
    model: str = "gpt-5.5",
    usage: CompletionUsage | None = None,
    choices: list[Choice] | None = None,
) -> ChatCompletionChunk:
    if choices is None:
        if delta is None:
            delta = ChoiceDelta()
        elif isinstance(delta, dict):
            delta = ChoiceDelta.model_validate(delta)
        choices = [Choice(index=0, delta=delta, finish_reason=finish_reason)]
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=0,
        model=model,
        choices=choices,
        usage=usage,
    )


def usage_chunk(prompt: int = 5, completion: int = 2, cached: int = 0):
    return chunk(
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
        ),
    )


def text_of(events: list[Event]) -> str:
    return "".join(
        data["delta"]["text"]
        for name, data in events
        if name == "content_block_delta"
        and data["delta"]["type"] == "text_delta"
    )


def assert_block_discipline(events: list[Event]) -> None:
    """Only one block open at a time, no delta against a closed or unopened
    block, every opened block closed, and nothing after `message_stop`."""
    open_index: int | None = None
    seen_indices: list[int] = []
    for position, (name, data) in enumerate(events):
        if name == "content_block_start":
            assert open_index is None, "a block was already open"
            open_index = data["index"]
            seen_indices.append(data["index"])
        elif name == "content_block_delta":
            assert open_index == data["index"], "delta against a closed block"
        elif name == "content_block_stop":
            assert open_index == data["index"], "stop for a non-open block"
            open_index = None
        elif name == "message_stop":
            assert position == len(events) - 1, "events follow message_stop"
    assert open_index is None, "a block was left open"
    # A single flat counter, monotonically increasing from 0.
    assert seen_indices == list(range(len(seen_indices)))


async def test_plain_text_stream():
    events = await translate(
        [
            chunk(ChoiceDelta(role="assistant", content="Hello")),
            chunk(ChoiceDelta(content=" world"), finish_reason="stop"),
            usage_chunk(),
        ]
    )
    assert [name for name, _ in events] == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert_block_discipline(events)
    assert events[0][1]["message"]["id"] == "chatcmpl_1"
    assert events[0][1]["message"]["stop_reason"] is None
    # `citations` rides along because the emitted content block is a real
    # `anthropic.types.TextBlock`, which declares that field.
    assert events[2][1]["content_block"] == {
        "type": "text",
        "text": "",
        "citations": None,
    }
    assert text_of(events) == "Hello world"

    message_delta = events[6][1]
    # `container`/`stop_details` ride along because the emitted `delta` is a
    # real `anthropic.types.raw_message_delta_event.Delta`.
    assert message_delta["delta"] == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "container": None,
        "stop_details": None,
    }
    # The emitted `usage` is a real `MessageDeltaUsage`, which declares more
    # fields than this translator populates, so only those four are asserted.
    assert (
        message_delta["usage"].items()
        >= {
            "input_tokens": 5,
            "output_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }.items()
    )


async def test_tool_call_stream():
    events = await translate(
        [
            chunk(
                ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name="get_weather", arguments=""
                            ),
                        )
                    ]
                )
            ),
            chunk(
                ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            function=ChoiceDeltaToolCallFunction(
                                arguments='{"city"'
                            ),
                        )
                    ]
                )
            ),
            chunk(
                ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            function=ChoiceDeltaToolCallFunction(
                                arguments=': "NYC"}'
                            ),
                        )
                    ]
                ),
                finish_reason="tool_calls",
            ),
            usage_chunk(),
        ]
    )
    assert_block_discipline(events)
    # `caller` rides along because the emitted block is a real
    # `anthropic.types.ToolUseBlock`.
    assert events[2][1]["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {},
        "caller": None,
    }
    partials = [
        data["delta"]["partial_json"]
        for name, data in events
        if name == "content_block_delta"
    ]
    assert partials == ['{"city"', ': "NYC"}']
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


async def test_an_aliased_tool_name_is_restored():
    aliases = ToolNameAliases()
    alias = aliases.to_upstream(LONG_MCP_NAME)
    events = await translate(
        [
            chunk(
                ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name=alias),
                        )
                    ]
                ),
                finish_reason="tool_calls",
            ),
            usage_chunk(),
        ],
        aliases=aliases,
    )
    start = next(d for name, d in events if name == "content_block_start")
    assert start["content_block"]["name"] == LONG_MCP_NAME


async def test_opening_a_tool_block_closes_the_open_text_block():
    events = await translate(
        [
            chunk(ChoiceDelta(content="thinking about it")),
            chunk(
                ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name="search"),
                        )
                    ]
                ),
                finish_reason="tool_calls",
            ),
            usage_chunk(),
        ]
    )
    assert_block_discipline(events)
    assert [name for name, _ in events].count("content_block_start") == 2


# --- §11.2 reasoning ---------------------------------------------------------


async def test_reasoning_stages_become_a_thinking_block():
    events = await translate(
        [
            chunk(
                {
                    "custom_content": {
                        "stages": [
                            {"index": 0, "name": "Thinking", "content": "let "}
                        ]
                    }
                }
            ),
            # A stage's name arrives only on its first delta.
            chunk(
                {"custom_content": {"stages": [{"index": 0, "content": "me"}]}}
            ),
            chunk(ChoiceDelta(content="answer"), finish_reason="stop"),
            usage_chunk(),
        ]
    )
    assert_block_discipline(events)
    start = events[2]
    assert start[0] == "content_block_start"
    assert start[1]["content_block"]["type"] == "thinking"
    thinking = "".join(
        data["delta"]["thinking"]
        for name, data in events
        if name == "content_block_delta"
        and data["delta"]["type"] == "thinking_delta"
    )
    assert thinking == "let me"
    assert text_of(events) == "answer"


async def test_a_non_reasoning_stage_is_ignored():
    events = await translate(
        [
            chunk(
                {
                    "custom_content": {
                        "stages": [
                            {"index": 0, "name": "Searching", "content": "x"}
                        ]
                    }
                }
            ),
            chunk(ChoiceDelta(content="answer"), finish_reason="stop"),
            usage_chunk(),
        ]
    )
    assert all(
        data.get("content_block", {}).get("type") != "thinking"
        for name, data in events
        if name == "content_block_start"
    )


async def test_a_signature_closes_the_thinking_block():
    events = await translate(
        [
            chunk(
                {
                    "custom_content": {
                        "stages": [
                            {"index": 0, "name": "Thinking", "content": "hmm"}
                        ]
                    }
                }
            ),
            chunk(
                {
                    "custom_content": {
                        "state": {
                            "claude_message_content": [
                                {
                                    "type": "thinking",
                                    "thinking": "hmm",
                                    "signature": "sig-abc",
                                }
                            ]
                        }
                    }
                }
            ),
            chunk(ChoiceDelta(content="answer"), finish_reason="stop"),
            usage_chunk(),
        ]
    )
    assert_block_discipline(events)
    names = [name for name, _ in events]
    signature_at = next(
        i
        for i, (name, data) in enumerate(events)
        if name == "content_block_delta"
        and data["delta"]["type"] == "signature_delta"
    )
    assert events[signature_at][1]["delta"]["signature"] == "sig-abc"
    # A signed block is complete.
    assert names[signature_at + 1] == "content_block_stop"


async def test_a_signature_after_the_block_closed_is_dropped_not_fatal():
    events = await translate(
        [
            chunk(ChoiceDelta(content="answer")),
            chunk(
                {
                    "custom_content": {
                        "state": {
                            "claude_message_content": [
                                {
                                    "type": "thinking",
                                    "thinking": "hmm",
                                    "signature": "sig",
                                }
                            ]
                        }
                    }
                },
                finish_reason="stop",
            ),
            usage_chunk(),
        ]
    )
    assert_block_discipline(events)
    assert text_of(events) == "answer"


# --- §11.2 citations ---------------------------------------------------------


def annotation(url: str, title: str = "T") -> dict:
    return {
        "type": "url_citation",
        "url_citation": {
            "url": url,
            "title": title,
            "start_index": 0,
            "end_index": 1,
        },
    }


async def test_annotations_become_citation_blocks():
    events = await translate(
        [
            chunk(ChoiceDelta(content="answer")),
            chunk({"annotations": [annotation("https://e.com")]}),
            chunk(ChoiceDelta(), finish_reason="stop"),
            usage_chunk(),
        ]
    )
    assert_block_discipline(events)
    types = [
        data["content_block"]["type"]
        for name, data in events
        if name == "content_block_start"
    ]
    assert types == ["text", "server_tool_use", "web_search_tool_result"]


async def test_a_resent_annotation_array_renders_each_citation_once():
    # `delta.annotations` is not uniformly delta-encoded: several adapters
    # resend the whole accumulated array on every chunk.
    events = await translate(
        [
            chunk({"annotations": [annotation("https://a.com")]}),
            chunk(
                {
                    "annotations": [
                        annotation("https://a.com"),
                        annotation("https://b.com"),
                    ]
                },
                finish_reason="stop",
            ),
            usage_chunk(),
        ]
    )
    assert_block_discipline(events)
    urls = [
        data["content_block"]["content"][0]["url"]
        for name, data in events
        if name == "content_block_start"
        and data["content_block"]["type"] == "web_search_tool_result"
    ]
    assert urls == ["https://a.com", "https://b.com"]


# --- §11.3 termination -------------------------------------------------------


async def test_usage_on_a_content_chunk_does_not_truncate():
    events = await translate(
        [
            chunk(ChoiceDelta(role="assistant", content="Hel")),
            chunk(
                ChoiceDelta(content="lo"),
                usage=CompletionUsage(
                    prompt_tokens=5, completion_tokens=1, total_tokens=6
                ),
            ),
            chunk(ChoiceDelta(content=" world"), finish_reason="stop"),
            usage_chunk(prompt=5, completion=2),
        ]
    )
    names = [name for name, _ in events]
    assert names.count("message_stop") == 1
    assert names[-1] == "message_stop"
    assert text_of(events) == "Hello world"
    message_delta = next(d for name, d in events if name == "message_delta")
    assert message_delta["usage"]["output_tokens"] == 2


async def test_content_after_the_terminal_usage_chunk_is_ignored():
    events = await translate(
        [
            chunk(ChoiceDelta(content="hi"), finish_reason="stop"),
            usage_chunk(),
            chunk(ChoiceDelta(content=" more")),
        ]
    )
    assert_block_discipline(events)
    assert text_of(events) == "hi"


async def test_stream_without_usage_chunk_still_finalizes():
    events = await translate(
        [
            chunk(ChoiceDelta(role="assistant", content="hi")),
            chunk(ChoiceDelta(), finish_reason="length"),
        ]
    )
    assert [name for name, _ in events][-1] == "message_stop"
    message_delta = next(d for name, d in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens"
    assert message_delta["usage"]["input_tokens"] == 0


async def test_an_upstream_yielding_no_chunks_still_emits_message_start():
    # A 200 with an empty body or a dropped connection still needs a
    # well-formed message: blocks with nothing to attach to break SDK parsers.
    events = await translate([])
    assert [name for name, _ in events] == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert_block_discipline(events)
    assert events[2][1]["content_block"] == {
        "type": "text",
        "text": "",
        "citations": None,
    }


async def test_a_stream_with_no_content_gets_the_zero_block_guard():
    events = await translate(
        [chunk(ChoiceDelta(), finish_reason="stop"), usage_chunk()]
    )
    assert_block_discipline(events)
    starts = [d for name, d in events if name == "content_block_start"]
    assert len(starts) == 1
    assert starts[0]["content_block"]["text"] == ""


async def test_cached_tokens_are_subtracted_in_streaming_too():
    events = await translate(
        [
            chunk(ChoiceDelta(content="hi"), finish_reason="stop"),
            usage_chunk(prompt=100, completion=20, cached=30),
        ]
    )
    usage = next(d for name, d in events if name == "message_delta")["usage"]
    assert usage["input_tokens"] == 70
    assert usage["cache_read_input_tokens"] == 30
    assert usage["cache_creation_input_tokens"] == 0


async def test_refusal_stream():
    events = await translate(
        [
            chunk(ChoiceDelta(role="assistant", refusal="I can't")),
            chunk(ChoiceDelta(), finish_reason="stop"),
            usage_chunk(),
        ]
    )
    assert text_of(events) == "I can't"
    assert events[-2][1]["delta"]["stop_reason"] == "refusal"


# --- §9 streaming stop-sequence emulation ------------------------------------


async def test_an_emulated_stop_sequence_truncates_the_stream():
    events = await translate(
        [
            chunk(ChoiceDelta(content="keep ST")),
            chunk(ChoiceDelta(content="OP drop")),
            chunk(ChoiceDelta(content=" more"), finish_reason="stop"),
            usage_chunk(),
        ],
        stop_sequences=["STOP"],
    )
    assert_block_discipline(events)
    assert text_of(events) == "keep "
    message_delta = next(d for name, d in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "stop_sequence"
    assert message_delta["delta"]["stop_sequence"] == "STOP"


async def test_a_withheld_tail_that_never_completes_is_released():
    events = await translate(
        [
            chunk(ChoiceDelta(content="hello ST"), finish_reason="stop"),
            usage_chunk(),
        ],
        stop_sequences=["STOP"],
    )
    assert text_of(events) == "hello ST"
    message_delta = next(d for name, d in events if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"


async def test_withheld_text_is_released_before_a_tool_block_opens():
    # Otherwise it would surface after the tool block and reorder the message.
    events = await translate(
        [
            chunk(ChoiceDelta(content="hello ST")),
            chunk(
                ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name="search"),
                        )
                    ]
                ),
                finish_reason="tool_calls",
            ),
            usage_chunk(),
        ],
        stop_sequences=["STOP"],
    )
    assert_block_discipline(events)
    block_types = [
        data["content_block"]["type"]
        for name, data in events
        if name == "content_block_start"
    ]
    assert block_types == ["text", "tool_use"]
    assert text_of(events) == "hello ST"


@pytest.mark.parametrize(
    "chunks, expected",
    [
        (["a", "b", "c"], "abc"),
        (["keep ", "STOP", " drop"], "keep "),
        (["k", "e", "e", "p", "S", "T", "O", "P"], "keep"),
    ],
)
async def test_streaming_text_matches_the_batch_matcher(chunks, expected):
    events = await translate(
        [chunk(ChoiceDelta(content=c)) for c in chunks]
        + [chunk(ChoiceDelta(), finish_reason="stop"), usage_chunk()],
        stop_sequences=["STOP"],
    )
    assert text_of(events) == expected
