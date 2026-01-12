from enum import Enum
from typing import Awaitable, Callable

from aidial_adapter_anthropic.llm.chat_model import ChatCompletionAdapter
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.storage import create_file_storage
from aidial_adapter_bedrock.llm.chat_model import default_preprocess_messages
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.default_tokenizer import (
    default_converse_tokenizer_factory,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDocumentType,
    ConverseImageType,
)
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.model.llama.v3 import (
    ConverseAdapterWithStreamingEmulation,
)


class ToolsSupport(Enum):
    NONE = 0
    NON_STREAMING_ONLY = 1
    ALWAYS = 2


class ConverseAdapterFactory(BaseModel):
    deployment: str
    get_client: Callable[[], Awaitable[Bedrock]]
    api_key: str

    async def create(
        self,
        *,
        tools_support: ToolsSupport = ToolsSupport.NONE,
        supported_image_types: list[ConverseImageType] | None = None,
        supported_document_types: list[ConverseDocumentType] | None = None,
        ensure_non_empty_tool_descriptions: bool = False,
    ) -> ChatCompletionAdapter:
        cls = (
            ConverseAdapterWithStreamingEmulation
            if tools_support == ToolsSupport.NON_STREAMING_ONLY
            else ConverseAdapter
        )

        model = cls(
            deployment=self.deployment,
            bedrock=await self.get_client(),
            storage=create_file_storage(self.api_key),
            input_tokenizer_factory=default_converse_tokenizer_factory,
            support_tools=tools_support != ToolsSupport.NONE,
            supported_image_types=supported_image_types or [],
            supported_document_types=supported_document_types or [],
            ensure_non_empty_tool_descriptions=ensure_non_empty_tool_descriptions,
        )
        return compose_decorators(
            preprocess_messages_decorator(default_preprocess_messages),
            replicator_decorator(),
        )(model)
