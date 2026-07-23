"""
Streaming translator: OpenAI Chat Completions events → Anthropic Messages SSE.
"""

from collections.abc import AsyncIterator

from anthropic.types import TextBlock, ToolUseBlock
from anthropic.types.input_json_delta import InputJSONDelta
from anthropic.types.text_delta import TextDelta
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from openai.types.completion_usage import CompletionUsage

from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    UNKNOWN_MESSAGE_ID,
    convert_usage,
    stop_reason,
)
from aidial_adapter_bedrock.anthropic_translator.sse_stream import (
    AnthropicStreamState,
    run_sse_stream,
)

_TEXT_KEY = "text"


class ChatCompletionsToAnthropicStream(AnthropicStreamState):
    def __init__(self, requested_model: str):
        super().__init__(requested_model, default_message_id=UNKNOWN_MESSAGE_ID)
        self.started = False
        self.finish_reason: str | None = None
        self.finished = False
        self.usage: CompletionUsage | None = None

    def handle(self, chunk: ChatCompletionChunk) -> list[bytes]:
        events: list[bytes] = []
        if not self.started:
            events.extend(self._on_start(chunk))

        if chunk.choices:
            events.extend(self._on_choice(chunk.choices[0]))

        # Remember the latest usage but don't finalize here: some upstreams
        # attach usage to a chunk that still has more content coming.
        if chunk.usage is not None:
            self.usage = chunk.usage

        return events

    def _on_start(self, chunk: ChatCompletionChunk) -> list[bytes]:
        self.started = True
        self.message_id = chunk.id or self.message_id
        self.model = chunk.model or self.requested_model
        return self.message_start_events()

    def _on_choice(self, choice: Choice) -> list[bytes]:
        events: list[bytes] = []
        delta = choice.delta

        if text := delta.content:
            events.extend(self._on_text_delta(text))
        if refusal := delta.refusal:
            self.saw_refusal = True
            events.extend(self._on_text_delta(refusal))
        for call in delta.tool_calls or []:
            events.extend(self._on_tool_call_delta(call))

        if choice.finish_reason:
            self.finish_reason = choice.finish_reason

        return events

    def _on_text_delta(self, text: str) -> list[bytes]:
        events: list[bytes] = []
        if _TEXT_KEY not in self.block_index:
            events.append(
                self._open_block(_TEXT_KEY, TextBlock(type="text", text=""))
            )
        events.extend(
            self._delta(_TEXT_KEY, TextDelta(type="text_delta", text=text))
        )
        return events

    def _on_tool_call_delta(self, call: ChoiceDeltaToolCall) -> list[bytes]:
        events: list[bytes] = []
        key: tuple[str, int] = ("tool", call.index)
        function = call.function
        if key not in self.block_index:
            events.append(
                self._open_block(
                    key,
                    ToolUseBlock(
                        type="tool_use",
                        id=call.id or "",
                        name=(function.name if function else None) or "",
                        input={},
                    ),
                )
            )
        if function and function.arguments:
            events.extend(
                self._delta(
                    key,
                    InputJSONDelta(
                        type="input_json_delta",
                        partial_json=function.arguments,
                    ),
                )
            )
        return events

    def finalize(self) -> list[bytes]:
        """Uses the last usage seen, or zeros if none arrived (e.g. the
        connection dropped or Core doesn't honour `stream_options`)."""
        if self.finished:
            return []
        self.finished = True
        return self._on_final(
            stop_reason(self.finish_reason, self.saw_refusal),
            convert_usage(self.usage),
        )


async def translate_stream(
    stream: AsyncStream[ChatCompletionChunk], requested_model: str
) -> AsyncIterator[bytes]:
    state = ChatCompletionsToAnthropicStream(requested_model)

    async for chunk in run_sse_stream(
        stream,
        state,
        state.handle,
        on_finalize=state.finalize,
        log_context="Chat Completions",
    ):
        yield chunk
