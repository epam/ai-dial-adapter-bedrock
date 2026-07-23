import json
from typing import Literal, cast

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage

from aidial_adapter_bedrock.anthropic_translator.chat_completions.streaming import (
    translate_stream,
)

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


class _FakeStream:
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


def _parse_anthropic_sse(raw: bytes) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
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


async def _translate(
    chunks: list[ChatCompletionChunk], model: str = "gpt-5.5"
) -> list[tuple[str, dict]]:
    stream = _FakeStream(chunks)
    collected = b""
    async for chunk in translate_stream(
        cast(AsyncStream[ChatCompletionChunk], stream), model
    ):
        collected += chunk
    return _parse_anthropic_sse(collected)


def _chunk(
    delta: ChoiceDelta | None = None,
    finish_reason: FinishReason | None = None,
    id: str = "chatcmpl_1",
    model: str = "gpt-5.5",
    usage: CompletionUsage | None = None,
    choices: list[Choice] | None = None,
) -> ChatCompletionChunk:
    if choices is None:
        choices = [
            Choice(
                index=0,
                delta=delta if delta is not None else ChoiceDelta(),
                finish_reason=finish_reason,
            )
        ]
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=0,
        model=model,
        choices=choices,
        usage=usage,
    )


async def test_plain_text_stream():
    result = await _translate(
        [
            _chunk(ChoiceDelta(role="assistant", content="Hello")),
            _chunk(ChoiceDelta(content=" world"), finish_reason="stop"),
            _chunk(
                choices=[],
                usage=CompletionUsage(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7
                ),
            ),
        ]
    )
    names = [name for name, _ in result]
    assert names == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert result[0][1]["message"]["id"] == "chatcmpl_1"
    # `citations` rides along because the emitted content block is now a real
    # `anthropic.types.TextBlock`, which declares that field (defaulting to
    # `None`) alongside `type`/`text`.
    assert result[2][1]["content_block"] == {
        "type": "text",
        "text": "",
        "citations": None,
    }
    assert result[3][1]["delta"] == {"type": "text_delta", "text": "Hello"}
    assert result[4][1]["delta"] == {"type": "text_delta", "text": " world"}

    message_delta = result[6][1]
    # `container`/`stop_details` ride along because the emitted `delta` is now
    # a real `anthropic.types.raw_message_delta_event.Delta`, which declares
    # those fields (defaulting to `None`) alongside `stop_reason`/
    # `stop_sequence`.
    assert message_delta["delta"] == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "container": None,
        "stop_details": None,
    }
    # `server_tool_use` rides along because the emitted `usage` is now a real
    # `anthropic.types.message_delta_usage.MessageDeltaUsage`, which declares
    # that field (defaulting to `None`) alongside the four this translator
    # populates.
    assert message_delta["usage"] == {
        "input_tokens": 5,
        "output_tokens": 2,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "server_tool_use": None,
    }


async def test_tool_call_stream():
    result = await _translate(
        [
            _chunk(
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
            _chunk(
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
            _chunk(
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
            _chunk(
                choices=[],
                usage=CompletionUsage(
                    prompt_tokens=8, completion_tokens=4, total_tokens=12
                ),
            ),
        ]
    )
    names = [name for name, _ in result]
    assert names == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    # `caller` rides along because the emitted content block is now a real
    # `anthropic.types.ToolUseBlock`, which declares that field (defaulting
    # to `None`) alongside `id`/`name`/`input`/`type`.
    assert result[2][1]["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {},
        "caller": None,
    }
    assert result[3][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"city"',
    }
    assert result[4][1]["delta"] == {
        "type": "input_json_delta",
        "partial_json": ': "NYC"}',
    }
    assert result[-2][1]["delta"]["stop_reason"] == "tool_use"


async def test_usage_on_a_content_chunk_does_not_truncate():
    # A chunk that carries (cumulative) usage while there is still more content
    # to come must NOT finalize the message early: all content must precede
    # the single terminal message_stop, and the final usage reflects the last
    # value seen.
    result = await _translate(
        [
            _chunk(ChoiceDelta(role="assistant", content="Hel")),
            _chunk(
                ChoiceDelta(content="lo"),
                usage=CompletionUsage(
                    prompt_tokens=5, completion_tokens=1, total_tokens=6
                ),
            ),
            _chunk(ChoiceDelta(content=" world"), finish_reason="stop"),
            _chunk(
                choices=[],
                usage=CompletionUsage(
                    prompt_tokens=5, completion_tokens=2, total_tokens=7
                ),
            ),
        ]
    )
    names = [name for name, _ in result]
    assert names.count("message_stop") == 1
    assert names[-1] == "message_stop"
    text = "".join(
        d["delta"]["text"]
        for name, d in result
        if name == "content_block_delta"
    )
    assert text == "Hello world"
    message_delta = next(d for name, d in result if name == "message_delta")
    assert message_delta["usage"]["output_tokens"] == 2


async def test_stream_without_usage_chunk_still_finalizes():
    # The stream ends after the finish_reason chunk without a trailing usage
    # chunk (e.g. the deployment doesn't honour include_usage); terminal
    # events are still emitted, with zeroed usage.
    result = await _translate(
        [
            _chunk(ChoiceDelta(role="assistant", content="hi")),
            _chunk(ChoiceDelta(), finish_reason="length"),
        ]
    )
    names = [name for name, _ in result]
    assert names[-1] == "message_stop"
    message_delta = next(d for name, d in result if name == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "max_tokens"
    assert message_delta["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "server_tool_use": None,
    }


async def test_refusal_stream():
    result = await _translate(
        [
            _chunk(ChoiceDelta(role="assistant", refusal="I can't")),
            _chunk(ChoiceDelta(), finish_reason="stop"),
            _chunk(
                choices=[],
                usage=CompletionUsage(
                    prompt_tokens=1, completion_tokens=1, total_tokens=2
                ),
            ),
        ]
    )
    names = [name for name, _ in result]
    assert "content_block_start" in names
    text_delta = next(d for name, d in result if name == "content_block_delta")
    assert text_delta["delta"] == {"type": "text_delta", "text": "I can't"}
    assert result[-2][1]["delta"]["stop_reason"] == "refusal"
