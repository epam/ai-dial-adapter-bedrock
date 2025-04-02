from dataclasses import dataclass
from logging import DEBUG
from typing import List, Literal, Optional, Tuple, Type, assert_never

from aidial_sdk.chat_completion import Message as DialMessage
from anthropic import NOT_GIVEN, NotGiven
from anthropic._resource import AsyncAPIResource
from anthropic.lib.bedrock import AsyncAnthropicBedrock
from anthropic.lib.streaming import (
    BetaContentBlockStopEvent as ContentBlockStopEvent,
)
from anthropic.lib.streaming import BetaInputJsonEvent as InputJsonEvent
from anthropic.lib.streaming import BetaTextEvent as TextEvent
from anthropic.lib.streaming._beta_types import (
    BetaCitationEvent as CitationEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaMessageStopEvent as MessageStopEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaSignatureEvent as SignatureEvent,
)
from anthropic.lib.streaming._beta_types import (
    BetaThinkingEvent as ThinkingEvent,
)
from anthropic.resources.beta import AsyncMessages as FirstPartyAsyncMessagesAPI
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaMessage as ClaudeResponseMessage
from anthropic.types.beta import BetaMessageParam as ClaudeMessageParam
from anthropic.types.beta import (
    BetaRawContentBlockDeltaEvent as ContentBlockDeltaEvent,
)
from anthropic.types.beta import (
    BetaRawContentBlockStartEvent as ContentBlockStartEvent,
)
from anthropic.types.beta import BetaRawMessageDeltaEvent as MessageDeltaEvent
from anthropic.types.beta import BetaRawMessageStartEvent as MessageStartEvent
from anthropic.types.beta import (
    BetaRedactedThinkingBlock as RedactedThinkingBlock,
)
from anthropic.types.beta import BetaTextBlock as TextBlock
from anthropic.types.beta import BetaThinkingBlock as ThinkingBlock
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from anthropic.types.beta import BetaToolChoiceAnyParam as ToolChoiceAnyParam
from anthropic.types.beta import BetaToolChoiceAutoParam as ToolChoiceAutoParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolChoiceToolParam as ToolChoiceToolParam
from anthropic.types.beta import BetaToolUseBlock as ToolUseBlock
from pydantic import BaseModel, Field

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
    MessageState,
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


# Beta AsyncMessages in Bedrock doesn't provide stream and count_tokens,
# so we enabled it via the adapter.
class _AsyncMessagesAdapter(AsyncAPIResource):
    create = FirstPartyAsyncMessagesAPI.create
    stream = FirstPartyAsyncMessagesAPI.stream

    # NOTE: count_tokens is still not supported by Bedrock.
    # The endpoint returns 200 {"Output":{"__type":"com.amazon.coral.service#UnknownOperationException"},"Version":"1.0"}
    # count_tokens = FirstPartyAsyncMessagesAPI.count_tokens

    def __init__(self, resource: AsyncAPIResource):
        super().__init__(resource._client)


# NOTE: it's not pydantic BaseModel, because
# anthropic.types.MessageParam.content is of Iterable type and
# pydantic automatically converts lists into
# list iterators following the type.
# See https://github.com/anthropics/anthropic-sdk-python/issues/656 for details.
@dataclass
class ClaudeRequest:
    params: ClaudeParameters
    messages: ListProjection[ClaudeMessageParam]


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
    type: Literal["enabled"]
    budget_tokens: int

    def to_claude(self) -> ThinkingConfigParam:
        return {"type": "enabled", "budget_tokens": self.budget_tokens}


class ThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"]

    def to_claude(self) -> ThinkingConfigParam:
        return {"type": "disabled"}


class BetaConfiguration(BaseModel):
    betas: List[AnthropicBetaParam] | None = Field(
        default=None,
        description="List of beta features to enable. Make sure to check if the given feature is supported by the Claude deployment you are using.",
    )


class ThinkingConfiguration(BetaConfiguration):
    # NOTE: once migrated to Pydantic v2 we could use TypeAdapter over
    # the anthropic's ThinkingConfigParam class directly.
    thinking: ThinkingConfigEnabled | ThinkingConfigDisabled | None = None


Configuration = BetaConfiguration | ThinkingConfiguration


class Adapter(ChatCompletionAdapter):
    deployment: AdapterDeployment[Claude3Deployment]
    storage: Optional[FileStorage]
    client: AsyncAnthropicBedrock

    @property
    def supports_thinking(self) -> bool:
        return (
            self.deployment.reference_deployment_id
            == ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET
        )

    async def configuration(self) -> Type[Configuration]:
        if self.supports_thinking:
            return ThinkingConfiguration
        else:
            return BetaConfiguration

    async def _parse_configuration(
        self, params: DialParameters
    ) -> Configuration:
        try:
            conf_cls = await self.configuration()
        except NotImplementedError:
            return BetaConfiguration()

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
            tools = [to_claude_tool_config(tool) for tool in tool_config.tools]

            match (tool_config.required, tool_config.tools):
                case (True, [tool]):
                    tool_choice = ToolChoiceToolParam(
                        type="tool", name=tool.function.name
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

        thinking: ThinkingConfigParam | NotGiven = NOT_GIVEN
        if (
            isinstance(configuration, ThinkingConfiguration)
            and configuration.thinking is not None
        ):
            thinking = configuration.thinking.to_claude()

        temperature = NOT_GIVEN
        if params.temperature is not None:
            # Mapping OpenAI temp [0,2] range to Anthropic temp [0,1] range
            temperature = params.temperature / 2

        if not isinstance(thinking, NotGiven) and thinking["type"] == "enabled":
            # Thinking isn’t compatible with temperature, top_p, or top_k
            # modifications as well as forced tool use.
            temperature = NOT_GIVEN

        claude_params = ClaudeParameters(
            max_tokens=params.max_tokens or DEFAULT_MAX_TOKENS_ANTHROPIC,
            stop_sequences=params.stop,
            system=system_prompt or NOT_GIVEN,
            temperature=temperature,
            top_p=params.top_p or NOT_GIVEN,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            betas=configuration.betas or NOT_GIVEN,
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

        async with (
            _AsyncMessagesAdapter(self.client.beta.messages).stream(
                messages=request.messages.raw_list,
                model=self.deployment.upstream_deployment_id,
                **request.params,
            ) as stream,
            consumer.create_stage("Thinking") as thinking_stage,
        ):
            dial_usage = TokenUsage()
            stop_reason = None

            async for event in stream:
                if log.isEnabledFor(DEBUG):
                    log.debug(f"response event: {json_dumps_short(event)}")

                match event:
                    case MessageStartEvent(message=message):
                        dial_usage.prompt_tokens += message.usage.input_tokens
                        dial_usage.cache_write_input_tokens += (
                            message.usage.cache_creation_input_tokens or 0
                        )
                        dial_usage.cache_read_input_tokens += (
                            message.usage.cache_read_input_tokens or 0
                        )
                    case TextEvent(text=text):
                        consumer.append_content(text)
                    case ThinkingEvent(thinking=thinking):
                        thinking_stage.append_content(thinking)
                    case SignatureEvent():
                        pass
                    case MessageDeltaEvent(usage=usage):
                        dial_usage.completion_tokens += usage.output_tokens
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
                                pass
                            case RedactedThinkingBlock():
                                pass
                            case _:
                                assert_never(content_block)
                    case MessageStopEvent(message=message):
                        stop_reason = message.stop_reason
                        if self.supports_thinking:
                            consumer.choice.set_state(
                                MessageState(
                                    claude_message_content=message.content
                                ).to_dict()
                            )
                    case (
                        InputJsonEvent()
                        | ContentBlockStartEvent()
                        | ContentBlockDeltaEvent()
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

            consumer.add_usage(dial_usage)

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

        message: ClaudeResponseMessage = await self.client.beta.messages.create(
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
                case ThinkingBlock(thinking=thinking):
                    with consumer.create_stage("Thinking") as stage:
                        stage.append_content(thinking)
                case RedactedThinkingBlock():
                    pass
                case _:
                    assert_never(content)

        if self.supports_thinking:
            consumer.choice.set_state(
                MessageState(claude_message_content=message.content).to_dict()
            )

        consumer.close_content(
            to_dial_finish_reason(message.stop_reason, tools_mode)
        )

        usage = message.usage
        dial_usage = TokenUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            cache_write_input_tokens=usage.cache_creation_input_tokens or 0,
        )

        consumer.add_usage(dial_usage)

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
