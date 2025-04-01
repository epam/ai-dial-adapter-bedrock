from logging import DEBUG
from typing import Any, Awaitable, Callable, List, Tuple

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
    extract_converse_system_prompt,
    to_converse_messages,
    to_converse_tools,
)
from aidial_adapter_bedrock.llm.converse.output import (
    process_non_streaming,
    process_streaming,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseDocumentType,
    ConverseImageType,
    ConverseMessage,
    ConverseRequestWrapper,
    ConverseTools,
    InferenceConfig,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    truncate_prompt,
)
from aidial_adapter_bedrock.utils.json import json_dumps_short, remove_nones
from aidial_adapter_bedrock.utils.list import omit_by_indices
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

ConverseMessages = List[Tuple[ConverseMessage, Any]]


class ConverseAdapter(ChatCompletionAdapter):
    deployment: str
    bedrock: Bedrock
    storage: FileStorage | None
    supported_image_types: list[ConverseImageType]
    supported_document_types: list[ConverseDocumentType]

    tokenize_text: Callable[[str], int] = default_tokenize_string
    input_tokenizer_factory: Callable[
        [ConverseDeployment, ConverseRequestWrapper],
        Callable[[ConverseMessages], Awaitable[int]],
    ]
    support_tools: bool
    partitioner: Callable[[ConverseMessages], List[int]] = (
        turn_based_partitioner
    )

    async def _discard_messages(
        self, params: ConverseRequestWrapper, max_prompt_tokens: int | None
    ) -> Tuple[DiscardedMessages | None, ConverseRequestWrapper]:
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
        ), ConverseRequestWrapper(
            messages=ListProjection(
                omit_by_indices(messages, discarded_messages)
            ),
            system=params.system,
            inferenceConfig=params.inferenceConfig,
            toolConfig=params.toolConfig,
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
    ) -> ConverseRequestWrapper:
        system_prompt_extraction = extract_converse_system_prompt(messages)
        converse_messages = await to_converse_messages(
            system_prompt_extraction.non_system_messages,
            self.storage,
            start_offset=system_prompt_extraction.system_message_count,
            supported_image_types=self.supported_image_types,
            supported_document_types=self.supported_document_types,
        )
        system_messages = system_prompt_extraction.system_messages
        if not converse_messages.list:
            raise ValidationError("List of messages must not be empty")

        return ConverseRequestWrapper(
            system=system_messages or None,
            messages=converse_messages,
            inferenceConfig=InferenceConfig(
                **remove_nones(
                    {
                        "temperature": params.temperature,
                        "topP": params.top_p,
                        "maxTokens": params.max_tokens,
                        "stopSequences": params.stop,
                    }
                )
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

        request = converse_params.to_request()

        if log.isEnabledFor(DEBUG):
            msg = json_dumps_short(
                {"deployment": self.deployment, "request": request}
            )
            log.debug(f"request: {msg}")

        if self.is_stream(params):
            await process_streaming(
                params=params,
                stream=(
                    await self.bedrock.aconverse_streaming(
                        self.deployment, **request
                    )
                ),
                consumer=consumer,
            )
        else:
            process_non_streaming(
                params=params,
                response=await self.bedrock.aconverse_non_streaming(
                    self.deployment, **request
                ),
                consumer=consumer,
            )
