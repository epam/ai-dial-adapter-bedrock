from logging import DEBUG
from typing import List, Optional, TypedDict, Union, assert_never

from aidial_sdk.chat_completion import Message
from anthropic import NOT_GIVEN, MessageStopEvent, NotGiven
from anthropic.lib.bedrock import AsyncAnthropicBedrock
from anthropic.lib.streaming import (
    AsyncMessageStream,
    InputJsonEvent,
    TextEvent,
)
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageParam,
    MessageStartEvent,
    MessageStreamEvent,
    TextBlock,
    ToolParam,
    ToolUseBlock,
)

from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import parse_dial_message
from aidial_adapter_bedrock.llm.model.claude.v3.converters import (
    ClaudeFinishReason,
    to_claude_messages,
    to_claude_tool_config,
    to_dial_finish_reason,
)
from aidial_adapter_bedrock.llm.model.claude.v3.tools import (
    process_tools_block,
    process_with_tools,
)
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode
from aidial_adapter_bedrock.utils.json import json_dumps_short
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
    tools: Union[List[ToolParam], NotGiven]


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

        tools = NOT_GIVEN
        tools_mode = None
        if params.tool_config is not None:
            tools = [
                to_claude_tool_config(tool_function)
                for tool_function in params.tool_config.functions
            ]
            tools_mode = params.tool_config.tools_mode

        parsed_messages = [
            process_with_tools(parse_dial_message(m), tools_mode)
            for m in messages
        ]

        system_prompt, claude_messages = await to_claude_messages(
            parsed_messages, self.storage
        )

        completion_params = ChatParams(
            max_tokens=params.max_tokens or DEFAULT_MAX_TOKENS_ANTHROPIC,
            stop_sequences=params.stop,
            system=system_prompt or NOT_GIVEN,
            temperature=(
                NOT_GIVEN
                if params.temperature is None
                else params.temperature / 2
            ),
            top_p=params.top_p or NOT_GIVEN,
            tools=tools,
        )

        if params.stream:
            await self.invoke_streaming(
                consumer, claude_messages, completion_params, tools_mode
            )
        else:
            await self.invoke_non_streaming(
                consumer, claude_messages, completion_params, tools_mode
            )

    async def invoke_streaming(
        self,
        consumer: Consumer,
        messages: List[MessageParam],
        params: ChatParams,
        tools_mode: ToolsMode | None,
    ):

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {"messages": messages, "model": self.model, "params": params}
            )
            log.debug(f"Streaming request: {msg}")

        async with self.client.messages.stream(
            messages=messages,
            model=self.model,
            **params,
        ) as stream:
            prompt_tokens = 0
            completion_tokens = 0
            stop_reason = None
            async for event in stream:
                match event:
                    case MessageStartEvent():
                        prompt_tokens += event.message.usage.input_tokens
                    case TextEvent():
                        consumer.append_content(event.text)
                    case MessageDeltaEvent():
                        completion_tokens += event.usage.output_tokens
                    case ContentBlockStopEvent():
                        if isinstance(event.content_block, ToolUseBlock):
                            process_tools_block(
                                consumer, event.content_block, tools_mode
                            )
                    case MessageStopEvent():
                        completion_tokens += event.message.usage.output_tokens
                        stop_reason = event.message.stop_reason
                    case (
                        InputJsonEvent()
                        | ContentBlockStartEvent()
                        | ContentBlockDeltaEvent()
                    ):
                        pass
                    case _:
                        raise ValueError(
                            f"Unsupported event type! {type(event)}"
                        )

            consumer.close_content(
                to_dial_finish_reason(stop_reason, tools_mode)
            )

            consumer.add_usage(
                TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )

    async def invoke_non_streaming(
        self,
        consumer: Consumer,
        messages: List[MessageParam],
        params: ChatParams,
        tools_mode: ToolsMode | None,
    ):

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {"messages": messages, "model": self.model, "params": params}
            )
            log.debug(f"Request: {msg}")

        message = await self.client.messages.create(
            messages=messages, model=self.model, **params, stream=False
        )
        for content in message.content:
            if isinstance(content, TextBlock):
                consumer.append_content(content.text)
            elif isinstance(content, ToolUseBlock):
                process_tools_block(consumer, content, tools_mode)
            else:
                assert_never(content)
        consumer.close_content(
            to_dial_finish_reason(message.stop_reason, tools_mode)
        )

        consumer.add_usage(
            TokenUsage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
            )
        )

    @classmethod
    def create(
        cls, model: str, api_key: str, aws_client_config: AWSClientConfig
    ):
        storage: Optional[FileStorage] = create_file_storage(api_key=api_key)
        client_kwargs = aws_client_config.get_anthropic_bedrock_client_kwargs()
        return cls(
            model=model,
            storage=storage,
            client=AsyncAnthropicBedrock(**client_kwargs),
        )
