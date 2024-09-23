from dataclasses import dataclass
from logging import DEBUG
from typing import List, Optional, Tuple, assert_never

from aidial_sdk.chat_completion import Message as DialMessage
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
)
from anthropic.types import MessageParam as ClaudeMessage
from anthropic.types import (
    MessageStartEvent,
    MessageStreamEvent,
    TextBlock,
    ToolUseBlock,
)
from anthropic.types.message_create_params import ToolChoice

from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
from aidial_adapter_bedrock.deployments import Claude3Deployment
from aidial_adapter_bedrock.dial_api.request import (
    ModelParameters as DialParameters,
)
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    keep_last,
    turn_based_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import parse_dial_message
from aidial_adapter_bedrock.llm.model.claude.v3.converters import (
    ClaudeFinishReason,
    to_claude_messages,
    to_claude_tool_config,
    to_dial_finish_reason,
)
from aidial_adapter_bedrock.llm.model.claude.v3.params import ClaudeParameters
from aidial_adapter_bedrock.llm.model.claude.v3.tokenizer import (
    create_tokenizer,
    tokenize_text,
)
from aidial_adapter_bedrock.llm.model.claude.v3.tools import (
    process_tools_block,
    process_with_tools,
)
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_ANTHROPIC
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)
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


# NOTE: it's not pydantic BaseModel, because
# ClaudeMessage.content is of Iterable type and
# pydantic automatically converts lists into
# list iterators following the type.
# See https://github.com/anthropics/anthropic-sdk-python/issues/656 for details.
@dataclass
class ClaudeRequest:
    params: ClaudeParameters
    messages: List[ClaudeMessage]


class Adapter(ChatCompletionAdapter):
    deployment: Claude3Deployment
    storage: Optional[FileStorage]
    client: AsyncAnthropicBedrock

    async def _prepare_claude_request(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> ClaudeRequest:
        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        tools = NOT_GIVEN
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN
        if params.tool_config is not None:
            tools = [
                to_claude_tool_config(tool_function)
                for tool_function in params.tool_config.functions
            ]
            tool_choice = (
                {"type": "any"}
                if params.tool_config.required
                else {"type": "auto"}
            )

        parsed_messages = [
            process_with_tools(parse_dial_message(m), params.tools_mode)
            for m in messages
        ]

        system_prompt, claude_messages = await to_claude_messages(
            parsed_messages, self.storage
        )

        claude_params = ClaudeParameters(
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
            tool_choice=tool_choice,
        )

        return ClaudeRequest(params=claude_params, messages=claude_messages)

    async def _compute_discarded_messages(
        self,
        request: ClaudeRequest,
        max_prompt_tokens: int | None,
    ) -> Tuple[DiscardedMessages | None, ClaudeRequest]:
        discarded_messages, messages = await truncate_prompt(
            messages=request.messages,
            tokenizer=create_tokenizer(self.deployment, request.params),
            keep_message=keep_last,
            partitioner=turn_based_partitioner,
            model_limit=None,
            user_limit=max_prompt_tokens,
        )

        if request.params["system"] is not NOT_GIVEN:
            discarded_messages = [idx + 1 for idx in discarded_messages]

        if max_prompt_tokens is None:
            discarded_messages = None

        return discarded_messages, ClaudeRequest(
            params=request.params, messages=messages
        )

    async def chat(
        self,
        consumer: Consumer,
        params: DialParameters,
        messages: List[DialMessage],
    ):
        request = await self._prepare_claude_request(params, messages)

        discarded_messages, request = await self._compute_discarded_messages(
            request, params.max_prompt_tokens
        )

        if params.stream:
            await self.invoke_streaming(
                consumer,
                params.tools_mode,
                request,
                discarded_messages,
            )
        else:
            await self.invoke_non_streaming(
                consumer,
                params.tools_mode,
                request,
                discarded_messages,
            )

    async def count_prompt_tokens(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> int:
        request = await self._prepare_claude_request(params, messages)
        return await create_tokenizer(self.deployment, request.params)(
            request.messages
        )

    async def count_completion_tokens(self, string: str) -> int:
        return tokenize_text(string)

    async def compute_discarded_messages(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> DiscardedMessages | None:
        request = await self._prepare_claude_request(params, messages)
        discarded_messages, _request = await self._compute_discarded_messages(
            request, params.max_prompt_tokens
        )
        return discarded_messages

    async def invoke_streaming(
        self,
        consumer: Consumer,
        tools_mode: ToolsMode | None,
        request: ClaudeRequest,
        discarded_messages: DiscardedMessages | None,
    ):

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {
                    "deployment": self.deployment,
                    "request": request,
                }
            )
            log.debug(f"Streaming request: {msg}")

        async with self.client.messages.stream(
            messages=request.messages,
            model=self.deployment.model_id,
            **request.params,
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

            consumer.set_discarded_messages(discarded_messages)

    async def invoke_non_streaming(
        self,
        consumer: Consumer,
        tools_mode: ToolsMode | None,
        request: ClaudeRequest,
        discarded_messages: DiscardedMessages | None,
    ):

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {
                    "deployment": self.deployment,
                    "request": request,
                }
            )
            log.debug(f"Request: {msg}")

        message = await self.client.messages.create(
            messages=request.messages,
            model=self.deployment.model_id,
            **request.params,
            stream=False,
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

        consumer.set_discarded_messages(discarded_messages)

    @classmethod
    def create(
        cls,
        deployment: Claude3Deployment,
        api_key: str,
        aws_client_config: AWSClientConfig,
    ):
        storage: Optional[FileStorage] = create_file_storage(api_key=api_key)
        client_kwargs = aws_client_config.get_anthropic_bedrock_client_kwargs()
        return cls(
            deployment=deployment,
            storage=storage,
            client=AsyncAnthropicBedrock(**client_kwargs),
        )
