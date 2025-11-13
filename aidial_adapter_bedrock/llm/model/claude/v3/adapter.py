from dataclasses import dataclass
from logging import DEBUG
from typing import List, Optional, Tuple, Type, assert_never

from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import ToolChoice as DialToolChoice
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, Omit, omit
from anthropic._resource import AsyncAPIResource
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
from anthropic.types.beta import (
    BetaBashCodeExecutionToolResultBlock as BashCodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaCodeExecutionToolResultBlock as CodeExecutionToolResultBlock,
)
from anthropic.types.beta import (
    BetaContainerUploadBlock as ContainerUploadBlock,
)
from anthropic.types.beta import BetaMCPToolResultBlock as MCPToolResultBlock
from anthropic.types.beta import BetaMCPToolUseBlock as MCPToolUseBlock
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
from anthropic.types.beta import BetaServerToolUseBlock as ServerToolUseBlock
from anthropic.types.beta import BetaTextBlock as TextBlock
from anthropic.types.beta import (
    BetaTextEditorCodeExecutionToolResultBlock as TextEditorCodeExecutionToolResultBlock,
)
from anthropic.types.beta import BetaThinkingBlock as ThinkingBlock
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from anthropic.types.beta import BetaToolChoiceAnyParam as ToolChoiceAnyParam
from anthropic.types.beta import BetaToolChoiceAutoParam as ToolChoiceAutoParam
from anthropic.types.beta import BetaToolChoiceNoneParam as ToolChoiceNoneParam
from anthropic.types.beta import BetaToolChoiceParam as ToolChoice
from anthropic.types.beta import BetaToolChoiceToolParam as ToolChoiceToolParam
from anthropic.types.beta import BetaToolUseBlock as ToolUseBlock
from anthropic.types.beta import (
    BetaWebFetchToolResultBlock as WebFetchToolResultBlock,
)
from anthropic.types.beta import (
    BetaWebSearchToolResultBlock as WebSearchToolResultBlock,
)

from aidial_adapter_bedrock.adapter_deployments import AdapterDeployment
from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from aidial_adapter_bedrock.deployments import ClaudeDeployment
from aidial_adapter_bedrock.dial_api.request import (
    ModelParameters as DialParameters,
)
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    default_preprocess_messages,
    keep_last,
    turn_based_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer, ToolUseMessage
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import parse_dial_message
from aidial_adapter_bedrock.llm.model.attachment_processor import (
    AttachmentProcessors,
)
from aidial_adapter_bedrock.llm.model.claude.v3.blocks import (
    IMAGE_ATTACHMENT_PROCESSOR,
    PDF_ATTACHMENT_PROCESSOR,
    TEXT_ATTACHMENT_PROCESSOR,
    create_text_block,
)
from aidial_adapter_bedrock.llm.model.claude.v3.config import (
    ClaudeConfiguration,
    ClaudeConfigurationWithThinking,
)
from aidial_adapter_bedrock.llm.model.claude.v3.converters import (
    to_claude_messages,
    to_claude_tool_config,
    to_dial_finish_reason,
    to_dial_usage,
)
from aidial_adapter_bedrock.llm.model.claude.v3.params import ClaudeParameters
from aidial_adapter_bedrock.llm.model.claude.v3.state import MessageState
from aidial_adapter_bedrock.llm.model.claude.v3.tokenizer import (
    create_tokenizer,
    tokenize_text,
)
from aidial_adapter_bedrock.llm.model.claude.v3.tools import (
    function_to_tool_messages,
    process_tools_block,
)
from aidial_adapter_bedrock.llm.model.conf import CLAUDE_DEFAULT_MAX_TOKENS
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)
from aidial_adapter_bedrock.upstream_config import UpstreamConfig
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


async def create_adapter(
    deployment: AdapterDeployment[ClaudeDeployment],
    api_key: str,
    upstream_config: UpstreamConfig,
) -> ChatCompletionAdapter:
    model = await Adapter.create(deployment, api_key, upstream_config)
    return compose_decorators(
        preprocess_messages_decorator(default_preprocess_messages),
        replicator_decorator(),
    )(model)


class Adapter(ChatCompletionAdapter):
    deployment: AdapterDeployment[ClaudeDeployment]
    storage: Optional[FileStorage]
    client: AsyncAnthropicBedrock | AsyncAnthropic

    @property
    def supports_thinking(self) -> bool:
        return self.deployment.reference_deployment_id in {
            D.ANTHROPIC_CLAUDE_V3_7_SONNET,
            D.ANTHROPIC_CLAUDE_V4_OPUS,
            D.ANTHROPIC_CLAUDE_V4_1_OPUS,
            D.ANTHROPIC_CLAUDE_V4_SONNET,
            D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
            D.ANTHROPIC_CLAUDE_V4_5_SONNET,
        }

    async def configuration(self) -> Type[ClaudeConfiguration]:
        if self.supports_thinking:
            return ClaudeConfigurationWithThinking
        return ClaudeConfiguration

    @property
    def attachment_processors(self) -> AttachmentProcessors:
        # Document support: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support#supported-platforms-and-models
        supports_documents = self.deployment.reference_deployment_id in {
            D.ANTHROPIC_CLAUDE_V3_5_HAIKU,
            D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
            D.ANTHROPIC_CLAUDE_V3_5_SONNET,
            D.ANTHROPIC_CLAUDE_V3_7_SONNET,
            D.ANTHROPIC_CLAUDE_V4_OPUS,
            D.ANTHROPIC_CLAUDE_V4_SONNET,
            D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
            D.ANTHROPIC_CLAUDE_V4_5_SONNET,
        }

        return AttachmentProcessors(
            text_handler=create_text_block,
            attachment_processors=(
                [IMAGE_ATTACHMENT_PROCESSOR]
                + (
                    [PDF_ATTACHMENT_PROCESSOR, TEXT_ATTACHMENT_PROCESSOR]
                    if supports_documents
                    else []
                )
            ),
            file_storage=self.storage,
        )

    async def _prepare_claude_request(
        self, params: DialParameters, messages: List[DialMessage]
    ) -> ClaudeRequest:
        configuration = params.parse_configuration(await self.configuration())

        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        tools = omit
        tool_choice: ToolChoice | Omit = omit
        if (
            tool_config := params.tool_config
        ) is not None and tool_config.tools:
            tools = [to_claude_tool_config(tool) for tool in tool_config.tools]

            match tool_config.tool_choice:
                case DialToolChoice(function=function):
                    tool_choice = ToolChoiceToolParam(
                        type="tool", name=function.name
                    )
                case "required":
                    tool_choice = ToolChoiceAnyParam(type="any")
                case "auto":
                    tool_choice = ToolChoiceAutoParam(type="auto")
                case "none":
                    tool_choice = ToolChoiceNoneParam(type="none")
                case _:
                    assert_never(tool_config.tool_choice)

            # NOTE tool_choice.disable_parallel_tool_use=True option isn't supported
            # by older Claude3 versions, so we limit the number of generated function calls
            # to one in the adapter itself for the functions mode.

        parsed_messages = [
            function_to_tool_messages(parse_dial_message(m)) for m in messages
        ]

        system_prompt, claude_messages = await to_claude_messages(
            self.attachment_processors, parsed_messages
        )

        thinking: ThinkingConfigParam | Omit = omit
        if (
            isinstance(configuration, ClaudeConfigurationWithThinking)
            and configuration.thinking is not None
        ):
            thinking = configuration.thinking.to_claude()

        temperature = omit
        if params.temperature is not None:
            # Mapping OpenAI temp [0,2] range to Anthropic temp [0,1] range
            temperature = params.temperature / 2

        if not isinstance(thinking, Omit) and thinking["type"] == "enabled":
            # Thinking isn’t compatible with temperature, top_p, or top_k
            # modifications as well as forced tool use.
            temperature = omit

        if (max_tokens := params.max_tokens) is None:
            max_tokens = CLAUDE_DEFAULT_MAX_TOKENS

        claude_params = ClaudeParameters(
            max_tokens=max_tokens,
            stop_sequences=params.stop,
            system=system_prompt or omit,
            temperature=temperature,
            top_p=params.top_p or omit,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            betas=configuration.betas or omit,
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
            stop_reason = None
            tool: ToolUseMessage | None = None

            async for event in stream:
                if log.isEnabledFor(DEBUG):
                    log.debug(f"response event: {json_dumps_short(event)}")

                match event:
                    case MessageStartEvent():
                        pass
                    case TextEvent(text=text):
                        consumer.append_content(text)

                    case ThinkingEvent(thinking=thinking):
                        thinking_stage.append_content(thinking)

                    case SignatureEvent() | MessageDeltaEvent():
                        pass

                    case ContentBlockStartEvent(content_block=content_block):
                        if isinstance(content_block, ToolUseBlock):
                            tool = process_tools_block(
                                consumer,
                                content_block,
                                tools_mode,
                                streaming=True,
                            )

                    case InputJsonEvent(partial_json=partial_json):
                        if tool:
                            tool.append_arguments(partial_json)
                        else:
                            log.warning(
                                "The model generated tool input before start using it"
                            )

                    case ContentBlockStopEvent(content_block=content_block):
                        match content_block:
                            case TextBlock():
                                # Already handled in TextEvent
                                pass
                            case ToolUseBlock():
                                # Tool Use is processed in ContentBlockStartEvent and InputJsonEvent handlers
                                pass
                            case ThinkingBlock() | RedactedThinkingBlock():
                                pass
                            case (
                                ServerToolUseBlock()
                                | WebSearchToolResultBlock()
                                | CodeExecutionToolResultBlock()
                                | MCPToolUseBlock()
                                | MCPToolResultBlock()
                                | ContainerUploadBlock()
                                | BashCodeExecutionToolResultBlock()
                                | TextEditorCodeExecutionToolResultBlock()
                                | WebFetchToolResultBlock()
                            ):
                                log.error(
                                    f"Content block of type {content_block.type} isn't supported"
                                )
                            case _:
                                assert_never(content_block)

                    case MessageStopEvent(message=message):
                        consumer.add_usage(to_dial_usage(message.usage))
                        stop_reason = message.stop_reason
                        if self.supports_thinking:
                            consumer.choice.set_state(
                                MessageState(
                                    claude_message_content=message.content
                                ).to_dict()
                            )

                    case ContentBlockDeltaEvent() | CitationEvent():
                        pass

                    case _:
                        assert_never(event)

            consumer.close_content(
                to_dial_finish_reason(stop_reason, tools_mode)
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
                    process_tools_block(
                        consumer, content, tools_mode, streaming=False
                    )
                case ThinkingBlock(thinking=thinking):
                    with consumer.create_stage("Thinking") as stage:
                        stage.append_content(thinking)
                case RedactedThinkingBlock():
                    pass
                case (
                    ServerToolUseBlock()
                    | WebSearchToolResultBlock()
                    | CodeExecutionToolResultBlock()
                    | MCPToolUseBlock()
                    | MCPToolResultBlock()
                    | ContainerUploadBlock()
                    | BashCodeExecutionToolResultBlock()
                    | TextEditorCodeExecutionToolResultBlock()
                    | WebFetchToolResultBlock()
                ):
                    log.error(
                        f"Content block of type {content} isn't supported"
                    )
                case _:
                    assert_never(content)

        if self.supports_thinking:
            consumer.choice.set_state(
                MessageState(claude_message_content=message.content).to_dict()
            )

        consumer.close_content(
            to_dial_finish_reason(message.stop_reason, tools_mode)
        )

        consumer.add_usage(to_dial_usage(message.usage))
        consumer.set_discarded_messages(discarded_messages)

    @classmethod
    async def create(
        cls,
        deployment: AdapterDeployment[ClaudeDeployment],
        api_key: str,
        upstream_config: UpstreamConfig,
    ):
        storage = create_file_storage(api_key=api_key)
        client = await create_anthropic_client(upstream_config)
        return cls(deployment=deployment, storage=storage, client=client)
