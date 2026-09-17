import pytest
from anthropic.types import Message
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    from_chat_completions,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.stop_emulation import (
    StopMatch,
    StopSequenceMatcher,
    apply_stop_sequences,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.streaming import (
    translate_stream,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from tests.unit_tests.anthropic_translator.helpers import (
    FakeStream,
    chunk,
    parse_anthropic_sse,
)


@pytest.mark.parametrize(
    "text,sequences,expected,matched",
    [
        ("abXYtail", ["abXY", "b"], "a", "b"),
        ("abtail", ["ab", "b"], "", "ab"),
        ("hello STOP tail", ["STOP"], "hello ", "STOP"),
        ("hello ST", ["STOP", ""], "hello ST", None),
        ("hello", [""], "hello", None),
    ],
)
def test_stop_matching_is_independent_of_chunk_boundaries(
    text: str, sequences: list[str], expected: str, matched: str | None
) -> None:
    result: StopMatch = apply_stop_sequences(text, sequences)
    assert (result.text, result.sequence) == (expected, matched)
    for split in range(len(text) + 1):
        matcher: StopSequenceMatcher = StopSequenceMatcher(sequences)
        visible: str = (
            matcher.push(text[:split])
            + matcher.push(text[split:])
            + matcher.flush()
        )
        assert (visible, matcher.matched) == (expected, matched)


def response(
    message: dict[str, object], usage: CompletionUsage | None = None
) -> ChatCompletion:
    return ChatCompletion(
        id="",
        model="",
        created=0,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage.model_validate(
                    {"role": "assistant", **message}
                ),
                finish_reason="length",
            )
        ],
        usage=usage,
    )


def test_non_stream_stop_removes_refusal_and_tools_but_keeps_usage() -> None:
    completion: ChatCompletion = response(
        {
            "content": "before STOP after",
            "refusal": "no",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ],
        },
        usage=CompletionUsage(
            prompt_tokens=7, completion_tokens=99, total_tokens=106
        ),
    )
    result: Message = from_chat_completions(
        completion, "requested", ToolNameAliases(), ["STOP"]
    )
    assert result.stop_reason == "stop_sequence"
    assert result.stop_sequence == "STOP"
    assert [block.model_dump()["text"] for block in result.content] == [
        "before "
    ]
    assert result.usage.output_tokens == 99
    assert result.id == "chatcmpl_unknown"
    assert result.model == "requested"


def test_non_stream_stops_do_not_span_content_and_refusal() -> None:
    result: Message = from_chat_completions(
        response({"content": "ST", "refusal": "OP"}),
        "m",
        ToolNameAliases(),
        ["STOP"],
    )
    assert result.stop_sequence is None
    assert len(result.content) == 2


async def events(
    chunks: list[ChatCompletionChunk], sequences: list[str] | None = None
):
    stream: FakeStream[ChatCompletionChunk] = FakeStream(chunks)
    raw: bytes = b"".join(
        [
            part
            async for part in translate_stream(
                stream,
                "m",
                ToolNameAliases(),
                sequences,
            )
        ]
    )
    assert stream.closed
    return [data for _, data in parse_anthropic_sse(raw)]


async def test_stream_stop_spans_deltas_and_consumes_final_usage() -> None:
    result = await events(
        [
            chunk({"content": "before ST"}),
            chunk(
                {
                    "content": "OP after",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call",
                            "function": {"name": "tool", "arguments": "{}"},
                        }
                    ],
                }
            ),
            chunk({"content": "discarded"}, finish_reason="length"),
            chunk(
                choices=[],
                usage=CompletionUsage(
                    prompt_tokens=7, completion_tokens=99, total_tokens=106
                ),
            ),
        ],
        ["STOP"],
    )
    assert (
        "".join(
            e["delta"]["text"]
            for e in result
            if e["type"] == "content_block_delta"
        )
        == "before "
    )
    assert result[-2]["delta"]["stop_reason"] == "stop_sequence"
    assert result[-2]["delta"]["stop_sequence"] == "STOP"
    assert result[-2]["usage"]["output_tokens"] == 99
    assert not any(
        e.get("content_block", {}).get("type") == "tool_use" for e in result
    )


async def test_stop_buffer_flushes_before_tool_transition() -> None:
    result = await events(
        [
            chunk({"content": "ST"}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call",
                            "function": {"name": "tool", "arguments": "{}"},
                        }
                    ]
                },
                finish_reason="tool_calls",
            ),
        ],
        ["STOP"],
    )
    assert result[3]["delta"]["text"] == "ST"
    assert result[-2]["delta"]["stop_reason"] == "tool_use"
