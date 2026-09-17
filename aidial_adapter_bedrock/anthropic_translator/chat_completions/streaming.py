from collections.abc import AsyncIterator

from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from anthropic.types.input_json_delta import InputJSONDelta
from anthropic.types.signature_delta import SignatureDelta
from anthropic.types.text_delta import TextDelta
from anthropic.types.thinking_delta import ThinkingDelta
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import AnnotationURLCitation
from openai.types.completion_usage import CompletionUsage

from aidial_adapter_bedrock.anthropic_translator.chat_completions.dial_extensions import (
    CustomContent,
    DialExtras,
    is_reasoning_stage,
    parse_extras,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    UNKNOWN_MESSAGE_ID,
    citation_blocks,
    convert_usage,
    stop_reason,
)
from aidial_adapter_bedrock.anthropic_translator.sse_stream import (
    AnthropicStreamState,
    ClosableAsyncStream,
    format_sse,
    run_sse_stream,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_TEXT_KEY: str = "text"
_THINKING_KEY: str = "thinking"
_CITATION_KEY: str = "citation"


class ChatCompletionsToAnthropicStream(AnthropicStreamState):
    def __init__(
        self,
        requested_model: str,
        aliases: ToolNameAliases,
    ) -> None:
        super().__init__(requested_model, default_message_id=UNKNOWN_MESSAGE_ID)
        self.started: bool = False
        self.finish_reason: str | None = None
        self.usage: CompletionUsage | None = None
        self.saw_refusal: bool = False
        self.saw_tool_use: bool = False
        self._aliases: ToolNameAliases = aliases
        self._reasoning_stages: set[int | None] = set()
        self._seen_citations: set[str] = set()
        self._signed: bool = False
        self._tools: dict[int, int] = {}

    def handle(self, chunk: ChatCompletionChunk) -> list[bytes]:
        if self.terminated:
            return []

        events: list[bytes] = []
        if not self.started:
            self.message_id = getattr(chunk, "id", None) or self.message_id
            self.model = getattr(chunk, "model", None) or self.requested_model
            events.extend(self._start())

        if chunk.choices:
            events.extend(self._on_choice(chunk.choices[0]))

        if chunk.usage is not None:
            self.usage = chunk.usage
            if not chunk.choices or self.finish_reason is not None:
                events.extend(self.finalize())

        return events

    def _start(self) -> list[bytes]:
        self.started = True
        return self.message_start_events()

    def _on_choice(self, choice: Choice) -> list[bytes]:
        events: list[bytes] = []
        delta: ChoiceDelta = choice.delta
        extras: DialExtras = parse_extras(delta.model_extra)

        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
        events.extend(self._on_custom_content(extras.custom_content))

        if text := delta.content:
            events.extend(self._emit_text(text))
        if refusal := delta.refusal:
            self.saw_refusal = True
            events.extend(self._emit_text(refusal))

        citations: list[tuple[str, str]] = []
        for annotation in extras.annotations or []:
            citation: AnnotationURLCitation = annotation.url_citation
            if citation.url and citation.url not in self._seen_citations:
                self._seen_citations.add(citation.url)
                citations.append((citation.url, citation.title or ""))
        if citations:
            events.extend(self._close_tools())
            for block in citation_blocks(citations):
                events.extend(self.open_block(_CITATION_KEY, block))
                events.extend(self.close_block())
        for call in delta.tool_calls or []:
            events.extend(self._on_tool_call_delta(call))

        return events

    def _on_custom_content(
        self, custom_content: CustomContent | None
    ) -> list[bytes]:
        if custom_content is None:
            return []

        events: list[bytes] = []
        for stage in custom_content.stages or []:
            if is_reasoning_stage(stage.name):
                self._reasoning_stages.add(stage.index)
            if stage.index in self._reasoning_stages and stage.content:
                events.extend(self._thinking_delta(stage.content))

        if self._signed or custom_content.state is None:
            return events
        for block in custom_content.state.claude_message_content or []:
            if block.type == "thinking" and block.signature:
                self._signed = True
                events.extend(self._signature_delta(block.signature))
                break
        return events

    def _thinking_delta(self, text: str) -> list[bytes]:
        events: list[bytes] = self._close_tools()
        if not self.is_open(_THINKING_KEY):
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
            log.warning("Thinking signature arrived after the block closed")
            return []

        events: list[bytes] = self.delta(
            SignatureDelta(type="signature_delta", signature=signature)
        )
        events.extend(self.close_block())
        return events

    def _emit_text(self, text: str) -> list[bytes]:
        if not text:
            return []
        events: list[bytes] = self._close_tools()
        if not self.is_open(_TEXT_KEY):
            events.extend(
                self.open_block(_TEXT_KEY, TextBlock(type="text", text=""))
            )
        events.extend(self.delta(TextDelta(type="text_delta", text=text)))
        return events

    def _on_tool_call_delta(self, call: ChoiceDeltaToolCall) -> list[bytes]:
        if call.type not in (None, "function"):
            return []
        events: list[bytes] = self.close_block()
        function: ChoiceDeltaToolCallFunction | None = call.function
        if call.index not in self._tools:
            self.saw_tool_use = True
            self._tools[call.index] = self.next_index
            self.next_index += 1
            events.append(
                format_sse(
                    RawContentBlockStartEvent(
                        type="content_block_start",
                        index=self._tools[call.index],
                        content_block=ToolUseBlock(
                            type="tool_use",
                            id=call.id or "",
                            name=self._aliases.to_client(
                                (function.name if function else None) or ""
                            ),
                            input={},
                        ),
                    )
                )
            )
        if function and function.arguments:
            events.append(
                format_sse(
                    RawContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=self._tools[call.index],
                        delta=InputJSONDelta(
                            type="input_json_delta",
                            partial_json=function.arguments,
                        ),
                    )
                )
            )
        return events

    def _close_tools(self) -> list[bytes]:
        events: list[bytes] = [
            format_sse(
                RawContentBlockStopEvent(type="content_block_stop", index=index)
            )
            for index in self._tools.values()
        ]
        self._tools.clear()
        return events

    def finalize(self) -> list[bytes]:
        if self.terminated:
            return []

        events: list[bytes] = []
        if not self.started:
            events.extend(self._start())
        events.extend(self._close_tools())
        if self.next_index == 0:
            events.extend(
                self.open_block(_TEXT_KEY, TextBlock(type="text", text=""))
            )

        events.extend(
            self.final_events(
                stop_reason(
                    self.finish_reason,
                    self.saw_tool_use,
                    self.saw_refusal,
                ),
                convert_usage(self.usage),
            )
        )
        return events


async def translate_stream(
    stream: ClosableAsyncStream[ChatCompletionChunk],
    requested_model: str,
    aliases: ToolNameAliases,
) -> AsyncIterator[bytes]:
    state: ChatCompletionsToAnthropicStream = ChatCompletionsToAnthropicStream(
        requested_model, aliases
    )

    async for chunk in run_sse_stream(
        stream,
        state,
        state.handle,
        on_finalize=state.finalize,
        log_context="Chat Completions",
    ):
        yield chunk
