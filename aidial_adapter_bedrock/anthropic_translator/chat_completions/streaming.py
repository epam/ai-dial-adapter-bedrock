"""
Streaming translator: OpenAI Chat Completions events → Anthropic Messages SSE.

The client asserts on an exact event sequence, reconstructed here from the
upstream's flat deltas:

    message_start -> ping
        -> (content_block_start -> delta* -> content_block_stop)*
        -> message_delta -> message_stop

Stop reason, usage arithmetic and citation blocks are shared verbatim with the
non-streaming path: a divergence between the two modes is a bug that manifests
in only one of them.
"""

from collections.abc import AsyncIterator

from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from anthropic.types.input_json_delta import InputJSONDelta
from anthropic.types.signature_delta import SignatureDelta
from anthropic.types.text_delta import TextDelta
from anthropic.types.thinking_delta import ThinkingDelta
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message import AnnotationURLCitation
from openai.types.completion_usage import CompletionUsage

from aidial_adapter_bedrock.anthropic_translator.chat_completions.dial_extensions import (
    CustomContent,
    is_reasoning_stage,
    parse_extras,
    signed_thinking,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    UNKNOWN_MESSAGE_ID,
    citation_blocks,
    convert_usage,
    stop_reason,
)
from aidial_adapter_bedrock.anthropic_translator.sse_stream import (
    AnthropicStreamState,
    run_sse_stream,
)
from aidial_adapter_bedrock.anthropic_translator.stop_sequences import (
    StopSequenceMatcher,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_TEXT_KEY = "text"
_THINKING_KEY = "thinking"
_CITATION_KEY = "citation"


class ChatCompletionsToAnthropicStream(AnthropicStreamState):
    def __init__(
        self,
        requested_model: str,
        aliases: ToolNameAliases,
        stop_sequences: list[str],
    ):
        super().__init__(requested_model, default_message_id=UNKNOWN_MESSAGE_ID)
        self.started = False
        self.finish_reason: str | None = None
        self.usage: CompletionUsage | None = None
        self.saw_refusal = False
        self.saw_tool_use = False
        self._aliases = aliases
        self._stop = StopSequenceMatcher(stop_sequences)
        self._reasoning_stages: set[int | None] = set()
        self._seen_citations: set[str] = set()
        self._signed = False

    def handle(self, chunk: ChatCompletionChunk) -> list[bytes]:
        if self.terminated:
            return []  # nothing may follow message_stop

        events: list[bytes] = []
        if not self.started:
            self.message_id = chunk.id or self.message_id
            self.model = chunk.model or self.requested_model
            events.extend(self._start())

        if chunk.choices:
            events.extend(self._on_choice(chunk.choices[0]))

        if chunk.usage is not None:
            # Remember it, but don't treat its arrival as terminal on its own:
            # some adapters attach usage to a chunk that still has content
            # coming, and ending there would stop the message mid-flight.
            self.usage = chunk.usage
            if not chunk.choices or self.finish_reason is not None:
                events.extend(self.finalize())

        return events

    def _start(self) -> list[bytes]:
        self.started = True
        return self.message_start_events()

    def _on_choice(self, choice: Choice) -> list[bytes]:
        events: list[bytes] = []
        delta = choice.delta
        extras = parse_extras(delta.model_extra)

        # Thinking leads the content array, so it is dispatched first.
        events.extend(self._on_custom_content(extras.custom_content))

        if text := delta.content:
            events.extend(self._emit_text(self._stop.push(text)))
        if refusal := delta.refusal:
            self.saw_refusal = True
            events.extend(self._emit_text(refusal))

        for annotation in extras.annotations or []:
            events.extend(self._on_annotation(annotation.url_citation))

        for call in delta.tool_calls or []:
            events.extend(self._on_tool_call_delta(call))

        if choice.finish_reason:
            # `stop_reason` belongs on the terminal `message_delta`, not inline.
            self.finish_reason = choice.finish_reason

        return events

    def _on_custom_content(
        self, custom_content: CustomContent | None
    ) -> list[bytes]:
        if custom_content is None:
            return []

        events: list[bytes] = []
        for stage in custom_content.stages or []:
            # A stage's name arrives only on its first delta, so which index is
            # the reasoning stage is recorded and later deltas keyed off it.
            if is_reasoning_stage(stage.name):
                self._reasoning_stages.add(stage.index)
            if stage.index in self._reasoning_stages and stage.content:
                events.extend(self._thinking_delta(stage.content))

        block = signed_thinking(custom_content)
        if not self._signed and block and block.signature:
            self._signed = True
            events.extend(self._signature_delta(block.signature))
        return events

    def _thinking_delta(self, text: str) -> list[bytes]:
        events: list[bytes] = []
        if not self.is_open(_THINKING_KEY):
            events.extend(self._flush_text())
            events.extend(
                self.open_block(
                    _THINKING_KEY,
                    ThinkingBlock(type="thinking", thinking="", signature=""),
                )
            )
        events.extend(
            self.delta(ThinkingDelta(type="thinking_delta", thinking=text))
        )
        return events

    def _signature_delta(self, signature: str) -> list[bytes]:
        if not self.is_open(_THINKING_KEY):
            # A delta cannot be sent to a closed block, so the client loses the
            # ability to replay this one.
            log.warning("Thinking signature arrived after the block closed")
            return []
        # A signed block is complete.
        events = self.delta(
            SignatureDelta(type="signature_delta", signature=signature)
        )
        events.extend(self.close_block())
        return events

    def _emit_text(self, text: str) -> list[bytes]:
        if not text:
            return []
        events: list[bytes] = []
        if not self.is_open(_TEXT_KEY):
            events.extend(
                self.open_block(_TEXT_KEY, TextBlock(type="text", text=""))
            )
        events.extend(self.delta(TextDelta(type="text_delta", text=text)))
        return events

    def _flush_text(self) -> list[bytes]:
        """Release text the stop matcher withheld before any other block can
        open, or it would surface after that block and reorder the message."""
        return self._emit_text(self._stop.flush())

    def _on_annotation(self, citation: AnnotationURLCitation) -> list[bytes]:
        # `delta.annotations` is not uniformly delta-encoded: several adapters
        # resend the whole accumulated array on every chunk.
        if not citation.url or citation.url in self._seen_citations:
            return []
        self._seen_citations.add(citation.url)

        events: list[bytes] = self._flush_text()
        # There is no delta form for a citation, so each block opens and closes.
        for block in citation_blocks(citation.url, citation.title or ""):
            events.extend(self.open_block(_CITATION_KEY, block))
            events.extend(self.close_block())
        return events

    def _on_tool_call_delta(self, call: ChoiceDeltaToolCall) -> list[bytes]:
        events: list[bytes] = []
        key: tuple[str, int] = ("tool", call.index)
        function = call.function
        if not self.is_open(key):
            self.saw_tool_use = True
            events.extend(self._flush_text())
            events.extend(
                self.open_block(
                    key,
                    ToolUseBlock(
                        type="tool_use",
                        id=call.id or "",
                        name=self._aliases.to_client(
                            (function.name if function else None) or ""
                        ),
                        input={},
                    ),
                )
            )
        if function and function.arguments:
            events.extend(
                self.delta(
                    InputJSONDelta(
                        type="input_json_delta",
                        partial_json=function.arguments,
                    )
                )
            )
        return events

    def finalize(self) -> list[bytes]:
        """Uses the last usage seen, or zeros if none arrived (e.g. the
        connection dropped or Core doesn't honour `stream_options`)."""
        if self.terminated:
            return []

        events: list[bytes] = []
        if not self.started:
            # An upstream that yielded no chunks at all — a 200 with an empty
            # body, a dropped connection — still needs a message to attach to.
            events.extend(self._start())
        events.extend(self._flush_text())
        if self.next_index == 0:
            # The zero-block guard, in streaming form.
            events.extend(
                self.open_block(_TEXT_KEY, TextBlock(type="text", text=""))
            )

        events.extend(
            self.final_events(
                stop_reason(
                    self.finish_reason,
                    self._stop.matched,
                    self.saw_tool_use,
                    self.saw_refusal,
                ),
                self._stop.matched,
                convert_usage(self.usage),
            )
        )
        return events


async def translate_stream(
    stream: AsyncStream[ChatCompletionChunk],
    requested_model: str,
    aliases: ToolNameAliases,
    stop_sequences: list[str],
) -> AsyncIterator[bytes]:
    state = ChatCompletionsToAnthropicStream(
        requested_model, aliases, stop_sequences
    )

    async for chunk in run_sse_stream(
        stream,
        state,
        state.handle,
        on_finalize=state.finalize,
        log_context="Chat Completions",
    ):
        yield chunk
