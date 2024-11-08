from typing import Any, Awaitable, Callable, List, Tuple, cast

from aidial_sdk.chat_completion import Message as DialMessage

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    keep_last,
    turn_based_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.converse.input import (
    get_converse_system_prompt,
    process_messages,
    to_converse_tools,
)
from aidial_adapter_bedrock.llm.converse.output import (
    process_non_streaming,
    process_streaming,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseMessage,
    ConverseParams,
    ConverseTools,
    InferenceConfig,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
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

    tokenize_text: Callable[[str], int] = default_tokenize_string
    input_tokenizer_factory: Callable[
        [ConverseDeployment, ConverseParams],
        Callable[[List[Tuple[ConverseMessage, Any]]], Awaitable[int]],
    ]
    support_tools: bool
    partitioner: Callable[[List[Tuple[ConverseMessage, Any]]], List[int]] = (
        turn_based_partitioner
    )

    async def _discard_messages(
        self, params: ConverseParams, max_prompt_tokens: int | None
    ) -> Tuple[DiscardedMessages | None, ConverseParams]:
        if max_prompt_tokens is None:
            return None, params

        discarded_messages, messages = await truncate_prompt(
            messages=params.messages.list,
            tokenizer=self.input_tokenizer_factory(self.deployment, params),
            keep_message=keep_last,
            partitioner=self.partitioner,
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
        return await self.input_tokenizer_factory(
            self.deployment, converse_params
        )(converse_params.messages.list)

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

    def get_tool_config(self, params: ModelParameters) -> ConverseTools | None:
        if params.tool_config and not self.support_tools:
            raise ValidationError("Tools are not supported")
        return (
            to_converse_tools(params.tool_config)
            if params.tool_config
            else None
        )

    async def construct_converse_params(
        self,
        messages: List[DialMessage],
        params: ModelParameters,
    ) -> ConverseParams:
        system_message = get_converse_system_prompt(messages)
        processed_messages = await process_messages(messages, self.storage)
        if not processed_messages.list:
            raise ValidationError("List of messages must not be empty")

        return ConverseParams(
            system=[system_message] if system_message else None,
            messages=processed_messages,
            inferenceConfig=cast(
                InferenceConfig,
                remove_nones(
                    {
                        "temperature": params.temperature,
                        "topP": params.top_p,
                        "maxTokens": params.max_tokens,
                        "stopSequences": params.stop,
                    }
                ),
            ),
            toolConfig=self.get_tool_config(params),
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
            await process_streaming(
                params=params,
                stream=(
                    await self.bedrock.aconverse_streaming(
                        self.deployment, **converse_params.to_dict()
                    )
                ),
                consumer=consumer,
            )
        else:
            process_non_streaming(
                params=params,
                response=await self.bedrock.aconverse_non_streaming(
                    self.deployment, **converse_params.to_dict()
                ),
                consumer=consumer,
            )
