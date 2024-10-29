import json
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Tuple,
    cast,
)

from aidial_sdk.chat_completion import FinishReason as DialFinishReason
from aidial_sdk.chat_completion import FunctionCall as DialFunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import ToolCall as DialToolCall

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    keep_last,
    turn_based_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.converse.input import (
    get_converse_system_prompt,
    process_messages,
    to_converse_finish_reason,
    to_converse_tools,
    to_dial_finish_reason,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseMessage,
    ConverseParams,
    InferenceConfig,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)
from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.list import omit_by_indices
from aidial_adapter_bedrock.utils.list_projection import ListProjection


class ConverseAdapter(ChatCompletionAdapter):
    deployment: str
    bedrock: Bedrock
    storage: FileStorage | None

    tokenize_text: Callable[[str], int]
    tokenizer_factory: Callable[
        [ConverseDeployment, ConverseParams],
        Callable[[List[Tuple[ConverseMessage, Any]]], Awaitable[int]],
    ]

    async def _process_streaming(
        self, stream: AsyncIterator[Any], consumer: Consumer
    ) -> None:
        current_tool_use = None

        async for event in stream:
            if (content_block_start := event.get("contentBlockStart")) and (
                tool_use := content_block_start.get("start", {}).get("toolUse")
            ):
                if current_tool_use is not None:
                    raise ValueError("Tool use already started")
                current_tool_use = {"input": ""} | tool_use

            elif content_block := event.get("contentBlockDelta"):
                delta = content_block.get("delta", {})

                if message := delta.get("text"):
                    consumer.append_content(message)

                if "toolUse" in delta:
                    if current_tool_use is None:
                        raise ValueError(
                            "Received tool delta before start block"
                        )
                    else:
                        current_tool_use["input"] += delta["toolUse"].get(
                            "input", ""
                        )

            elif event.get("contentBlockStop"):
                if current_tool_use:
                    consumer.create_function_tool_call(
                        tool_call=DialToolCall(
                            type="function",
                            id=current_tool_use["toolUseId"],
                            index=None,
                            function=DialFunctionCall(
                                name=current_tool_use["name"],
                                arguments=current_tool_use["input"],
                            ),
                        )
                    )
                    current_tool_use = None

            elif (message_stop := event.get("messageStop")) and (
                stop_reason := message_stop.get("stopReason")
            ):
                consumer.close_content(to_dial_finish_reason(stop_reason))

    async def _discard_messages(
        self, params: ConverseParams, max_prompt_tokens: int | None
    ) -> Tuple[DiscardedMessages | None, ConverseParams]:
        if max_prompt_tokens is None:
            return None, params

        discarded_messages, messages = await truncate_prompt(
            messages=params.messages.list,
            tokenizer=self.tokenizer_factory(self.deployment, params),
            keep_message=keep_last,
            partitioner=turn_based_partitioner,
            model_limit=None,
            user_limit=max_prompt_tokens,
        )

        return list(
            params.messages.to_original_indices(discarded_messages)
        ), ConverseParams(
            **{
                **params.to_dict(),
                "messages": ListProjection(
                    omit_by_indices(messages, discarded_messages)
                ),
            },
        )

    async def count_prompt_tokens(
        self, params: ModelParameters, messages: List[DialMessage]
    ) -> int:
        converse_params = await self.construct_converse_params(messages, params)
        return await self.tokenizer_factory(self.deployment, converse_params)(
            converse_params.messages.list
        )

    async def count_completion_tokens(self, string: str) -> int:
        return self.tokenize_text(string)

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[DialMessage]
    ) -> DiscardedMessages | None:
        converse_params = await self.construct_converse_params(messages, params)
        discarded_messages, _ = await self._discard_messages(
            converse_params, params.max_prompt_tokens
        )
        return discarded_messages

    def _process_non_streaming(
        self, response: Dict[str, Any], consumer: Consumer
    ) -> None:
        message = response["output"]["message"]
        for content_block in message.get("content", []):
            if "text" in content_block:
                consumer.append_content(content_block["text"])
            if "toolUse" in content_block:
                consumer.create_function_tool_call(
                    tool_call=DialToolCall(
                        type="function",
                        id=content_block["toolUse"]["toolUseId"],
                        index=None,
                        function=DialFunctionCall(
                            name=content_block["toolUse"]["name"],
                            arguments=json.dumps(
                                content_block["toolUse"]["input"]
                            ),
                        ),
                    )
                )

        if usage := response.get("usage"):
            consumer.add_usage(
                TokenUsage(
                    prompt_tokens=usage.get("inputTokens", 0),
                    completion_tokens=usage.get("outputTokens", 0),
                )
            )

        if stop_reason := response.get("stopReason"):
            consumer.close_content(to_dial_finish_reason(stop_reason))

    async def construct_converse_params(
        self,
        messages: List[DialMessage],
        params: ModelParameters,
    ) -> ConverseParams:
        system_message = get_converse_system_prompt(messages)
        return ConverseParams(
            system=[system_message] if system_message else None,
            messages=await process_messages(messages, self.storage),
            inferenceConfig=cast(
                InferenceConfig,
                remove_nones(
                    {
                        "temperature": params.temperature,
                        "topP": params.top_p,
                        "maxTokens": params.max_tokens,
                        "stopSequences": [
                            to_converse_finish_reason(
                                DialFinishReason(finish_reason)
                            ).value
                            for finish_reason in params.stop
                        ],
                    }
                ),
            ),
            toolConfig=(
                to_converse_tools(params.tool_config)
                if params.tool_config
                else None
            ),
        )

    def is_stream(self, params: ModelParameters) -> bool:
        return params.stream

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[DialMessage],
    ) -> None:

        converse_params = await self.construct_converse_params(messages, params)
        discarded_messages, converse_params = await self._discard_messages(
            converse_params, params.max_prompt_tokens
        )
        if not converse_params.messages.raw_list:
            raise ValidationError("No messages left after truncation")

        consumer.set_discarded_messages(discarded_messages)

        if self.is_stream(params):
            await self._process_streaming(
                stream=(
                    await self.bedrock.aconverse_streaming(
                        self.deployment, **converse_params.to_dict()
                    )
                ),
                consumer=consumer,
            )
        else:
            self._process_non_streaming(
                response=await self.bedrock.aconverse_non_streaming(
                    self.deployment, **converse_params.to_dict()
                ),
                consumer=consumer,
            )
