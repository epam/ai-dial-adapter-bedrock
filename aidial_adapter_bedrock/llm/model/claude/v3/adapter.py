from typing import List, Mapping, Optional, TypedDict, Union

from aidial_sdk.chat_completion import Message
from anthropic import NOT_GIVEN, NotGiven
from anthropic.lib.bedrock import AsyncAnthropicBedrock
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import (
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStreamEvent,
)

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.model.claude.v3.converters import (
    ClaudeFinishReason,
    to_claude_messages,
    to_dial_finish_reason,
)
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.llm.tools.claude_emulator import (
    legacy_tools_emulator,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class UsageEventHandler(AsyncMessageStream):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: Optional[ClaudeFinishReason] = None

    async def on_stream_event(self, event: MessageStreamEvent):
        if isinstance(event, MessageStartEvent):
            self.prompt_tokens = event.message.usage.input_tokens
        elif isinstance(event, MessageDeltaEvent):
            self.completion_tokens += event.usage.output_tokens
            self.stop_reason = event.delta.stop_reason


class ChatParams(TypedDict):
    max_tokens: int
    stop_sequences: Union[List[str], NotGiven]
    system: Union[str, NotGiven]
    temperature: Union[float, NotGiven]
    top_p: Union[float, NotGiven]


class Adapter(ChatCompletionAdapter):
    model: str
    storage: Optional[FileStorage]
    client: AsyncAnthropicBedrock

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ):
        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        tools_emulator = self.tools_emulator(params.tool_config)
        base_messages = tools_emulator.parse_dial_messages(messages)
        tool_stop_sequences = tools_emulator.get_stop_sequences()

        prompt, claude_messages = await to_claude_messages(
            base_messages, self.storage
        )

        completion_params = ChatParams(
            max_tokens=params.max_tokens or DEFAULT_MAX_TOKENS_ANTHROPIC,
            stop_sequences=[*params.stop, *tool_stop_sequences],
            system=prompt or NOT_GIVEN,
            temperature=(
                NOT_GIVEN
                if params.temperature is None
                else params.temperature / 2
            ),
            top_p=params.top_p or NOT_GIVEN,
        )

        if params.stream:
            await self.invoke_streaming(
                consumer, claude_messages, completion_params
            )
        else:
            await self.invoke_non_streaming(
                consumer, claude_messages, completion_params
            )

    async def invoke_streaming(
        self,
        consumer: Consumer,
        messages: List[MessageParam],
        params: ChatParams,
    ):
        log.debug(
            f"Streaming request: messages={messages}, model={self.model}, params={params}"
        )
        async with self.client.messages.stream(
            messages=messages,
            model=self.model,
            event_handler=UsageEventHandler,
            **params,
        ) as stream:
            async for text in stream.text_stream:
                consumer.append_content(text)
            consumer.close_content(to_dial_finish_reason(stream.stop_reason))

            consumer.add_usage(
                TokenUsage(
                    prompt_tokens=stream.prompt_tokens,
                    completion_tokens=stream.completion_tokens,
                )
            )

    async def invoke_non_streaming(
        self,
        consumer: Consumer,
        messages: List[MessageParam],
        params: ChatParams,
    ):
        log.debug(
            f"Request: messages={messages}, model={self.model}, params={params}"
        )
        message = await self.client.messages.create(
            messages=messages, model=self.model, **params, stream=False
        )
        prompt_tokens = 0
        completion_tokens = 0
        for content in message.content:
            usage = message.usage
            prompt_tokens = usage.input_tokens
            completion_tokens += usage.output_tokens
            consumer.append_content(content.text)
        consumer.close_content(to_dial_finish_reason(message.stop_reason))

        consumer.add_usage(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    @classmethod
    def create(cls, model: str, region: str, headers: Mapping[str, str]):
        storage: Optional[FileStorage] = create_file_storage(headers)
        return cls(
            model=model,
            tools_emulator=legacy_tools_emulator,
            storage=storage,
            client=AsyncAnthropicBedrock(aws_region=region),
        )
