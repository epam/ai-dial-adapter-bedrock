from dataclasses import dataclass
from logging import DEBUG
from typing import List, Literal, Optional, Tuple, Type, assert_never

from aidial_sdk.chat_completion import Message as DialMessage
from anthropic import NOT_GIVEN, MessageStopEvent, NotGiven
from anthropic.lib.bedrock import AsyncAnthropicBedrock
from anthropic.lib.streaming import (
    ContentBlockStopEvent,
    InputJsonEvent,
    TextEvent,
)
from anthropic.lib.streaming._types import (
    CitationEvent,
    SignatureEvent,
    ThinkingEvent,
)
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageDeltaEvent,
)
from anthropic.types import MessageParam as ClaudeMessage
from anthropic.types import (
    MessageStartEvent,
    TextBlock,
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceToolParam,
    ToolUseBlock,
)
from anthropic.types.message_create_params import ToolChoice
from anthropic.types.redacted_thinking_block import RedactedThinkingBlock
from anthropic.types.thinking_block import ThinkingBlock
from pydantic import BaseModel

from aidial_adapter_bedrock.adapter_deployments import AdapterDeployment
from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    Claude3Deployment,
)
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
    default_preprocess_messages,
    keep_last,
    turn_based_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import parse_dial_message
from aidial_adapter_bedrock.llm.model.claude.v3.converters import (
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
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


# NOTE: it's not pydantic BaseModel, because
# ClaudeMessage.content is of Iterable type and
# pydantic automatically converts lists into
# list iterators following the type.
# See https://github.com/anthropics/anthropic-sdk-python/issues/656 for details.
@dataclass
class ClaudeRequest:
    params: ClaudeParameters
    messages: ListProjection[ClaudeMessage]


def create_adapter(
    deployment: AdapterDeployment[Claude3Deployment],
    api_key: str,
    aws_client_config: AWSClientConfig,
) -> ChatCompletionAdapter:
    model = Adapter.create(deployment, api_key, aws_client_config)
    return compose_decorators(
        preprocess_messages_decorator(default_preprocess_messages),
        replicator_decorator(),
    )(model)


class ThinkingConfigEnabled(BaseModel):
    budget_tokens: int
    type: Literal["enabled"]


class ThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"]


class Configuration(BaseModel):
    # NOTE: once migrated to Pydantic v2 we could use TypeAdapter over
    # the anthropic's ThinkingConfigParam class directly.
    thinking: ThinkingConfigEnabled | ThinkingConfigDisabled | None = None


class Adapter(ChatCompletionAdapter):
    deployment: AdapterDeployment[Claude3Deployment]
    storage: Optional[FileStorage]
    client: AsyncAnthropicBedrock

    async def configuration(self) -> Type[Configuration]:
        if (
            self.deployment.reference_deployment_id
            == ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET_US
        ):
            return Configuration
        raise NotImplementedError

    async def _parse_configuration(
        self, params: DialParameters
    ) -> Configuration:
        try:
            conf_cls = await self.configuration()
        except NotImplementedError:
            return Configuration()

        return params.parse_configuration(conf_cls)

    async def _prepare_claude_request(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> ClaudeRequest:
        configuration = await self._parse_configuration(params)

        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        tools = NOT_GIVEN
        tool_choice: ToolChoice | NotGiven = NOT_GIVEN
        if (tool_config := params.tool_config) is not None:
            tools = [
                to_claude_tool_config(tool_function)
                for tool_function in tool_config.functions
            ]

            match (tool_config.required, tool_config.functions):
                case (True, [func]):
                    tool_choice = ToolChoiceToolParam(
                        type="tool", name=func.name
                    )
                case (True, _):
                    tool_choice = ToolChoiceAnyParam(type="any")
                case (False, _):
                    tool_choice = ToolChoiceAutoParam(type="auto")
                case _:
                    assert_never(tool_config)

            # NOTE tool_choice.disable_parallel_tool_use=True option isn't supported
            # by older Claude3 versions, so we limit the number of generated function calls
            # to one in the adapter itself for the functions mode.

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
            thinking=configuration.thinking.dict() or NOT_GIVEN,  # type: ignore
        )

        return ClaudeRequest(params=claude_params, messages=claude_messages)

    async def _compute_discarded_messages(
        self, request: ClaudeRequest, max_prompt_tokens: int | None
    ) -> Tuple[DiscardedMessages | None, ClaudeRequest]:
        if max_prompt_tokens is None:
            return None, request

        discarded_messages, messages = await truncate_prompt(
            messages=request.messages.list,
            tokenizer=create_tokenizer(
                self.deployment.reference_deployment_id, request.params
            ),
            keep_message=keep_last,
            partitioner=turn_based_partitioner,
            model_limit=None,
            user_limit=max_prompt_tokens,
        )

        claude_messages = ListProjection(messages)

        discarded_messages = list(
            request.messages.to_original_indices(discarded_messages)
        )

        return discarded_messages, ClaudeRequest(
            params=request.params,
            messages=claude_messages,
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
        return await create_tokenizer(
            self.deployment.reference_deployment_id, request.params
        )(request.messages.list)

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
                {"deployment": self.deployment, "request": request}
            )
            log.debug(f"request: {msg}")

        async with self.client.messages.stream(
            messages=request.messages.raw_list,
            model=self.deployment.upstream_deployment_id,
            **request.params,
        ) as stream:
            prompt_tokens = 0
            completion_tokens = 0
            stop_reason = None
            async for event in stream:
                if log.isEnabledFor(DEBUG):
                    log.debug(f"response event: {json_dumps_short(event)}")

                match event:
                    case MessageStartEvent(message=message):
                        prompt_tokens += message.usage.input_tokens
                    case TextEvent(text=text):
                        consumer.append_content(text)
                    case MessageDeltaEvent(usage=usage):
                        completion_tokens += usage.output_tokens
                    case ContentBlockStopEvent(content_block=content_block):
                        match content_block:
                            case ToolUseBlock():
                                process_tools_block(
                                    consumer, content_block, tools_mode
                                )
                            case TextBlock():
                                # Already handled in TextEvent
                                pass
                            case ThinkingBlock():
                                pass  # FIXME
                            case RedactedThinkingBlock():
                                pass  # FIXME
                            case _:
                                assert_never(content_block)
                    case MessageStopEvent(message=message):
                        stop_reason = message.stop_reason
                    case (
                        InputJsonEvent()
                        | ContentBlockStartEvent()
                        | ContentBlockDeltaEvent()
                        | ThinkingEvent()  # FIXME
                        | SignatureEvent()  # FIXME
                        # NOTE: the document understanding isn't supported in Bedrock yet:
                        # https://github.com/epam/ai-dial-adapter-bedrock/pull/227
                        | CitationEvent()
                    ):
                        pass
                    case _:
                        assert_never(event)

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
                {"deployment": self.deployment, "request": request}
            )
            log.debug(f"request: {msg}")

        message = await self.client.messages.create(
            messages=request.messages.raw_list,
            model=self.deployment.upstream_deployment_id,
            **request.params,
            stream=False,
        )

        if log.isEnabledFor(DEBUG):
            log.debug(f"response: {json_dumps_short(message)}")

        for content in message.content:
            match content:
                case TextBlock(text=text):
                    consumer.append_content(text)
                case ToolUseBlock():
                    process_tools_block(consumer, content, tools_mode)
                case ThinkingBlock():
                    pass  # FIXME
                case RedactedThinkingBlock():
                    pass  # FIXME
                case _:
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
        deployment: AdapterDeployment[Claude3Deployment],
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
