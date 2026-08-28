from collections.abc import Awaitable, Callable
from enum import Enum
from typing import assert_never

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as CCD
from aidial_adapter_bedrock.dial_api.storage import create_file_storage
from aidial_adapter_bedrock.llm.chat_model import default_preprocess_messages
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.tokenizers import (
    default_converse_tokenizer_factory,
    upstream_converse_tokenizer_factory,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseDocumentType,
    ConverseImageType,
    ConverseMessages,
    ConverseRequestWrapper,
)
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.caching import caching_decorator
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.model.llama.v3 import (
    ConverseAdapterWithStreamingEmulation,
)
from aidial_adapter_bedrock.utils.adapter_deployments import (
    AdapterChatCompletionDeployment,
)


class ToolsSupport(Enum):
    NONE = 0
    NON_STREAMING_ONLY = 1
    ALWAYS = 2


_TOKENIZER_FACTORY = Callable[
    [ConverseDeployment, Bedrock, ConverseRequestWrapper],
    Callable[[ConverseMessages], Awaitable[int]],
]


def _get_tokenizer_factory(
    deployment: CCD,
) -> _TOKENIZER_FACTORY:
    match deployment:
        case (
            CCD.AMAZON_NOVA_PRO
            | CCD.AMAZON_NOVA_LITE
            | CCD.AMAZON_NOVA_MICRO
            | CCD.AI21_JAMBA_1_5_LARGE_V1
            | CCD.AI21_JAMBA_1_5_MINI_V1
            | CCD.ANTHROPIC_CLAUDE_V3_SONNET
            | CCD.ANTHROPIC_CLAUDE_V3_5_SONNET
            | CCD.ANTHROPIC_CLAUDE_V3_5_SONNET_V2
            | CCD.ANTHROPIC_CLAUDE_V3_HAIKU
            | CCD.ANTHROPIC_CLAUDE_V3_5_HAIKU
            | CCD.ANTHROPIC_CLAUDE_V3_OPUS
            | CCD.ANTHROPIC_CLAUDE_V3_7_SONNET
            | CCD.ANTHROPIC_CLAUDE_V4_OPUS
            | CCD.ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE
            | CCD.ANTHROPIC_CLAUDE_V4_7_OPUS
            | CCD.ANTHROPIC_CLAUDE_V4_8_OPUS
            | CCD.ANTHROPIC_CLAUDE_V5_SONNET
            | CCD.ANTHROPIC_CLAUDE_V5_OPUS
            | CCD.META_LLAMA3_8B_INSTRUCT_V1
            | CCD.META_LLAMA3_70B_INSTRUCT_V1
            | CCD.META_LLAMA3_1_8B_INSTRUCT_V1
            | CCD.META_LLAMA3_1_70B_INSTRUCT_V1
            | CCD.META_LLAMA3_1_405B_INSTRUCT_V1
            | CCD.META_LLAMA3_2_1B_INSTRUCT_V1
            | CCD.META_LLAMA3_2_3B_INSTRUCT_V1
            | CCD.META_LLAMA3_2_11B_INSTRUCT_V1
            | CCD.META_LLAMA3_2_90B_INSTRUCT_V1
            | CCD.META_LLAMA3_3_70B_INSTRUCT_V1
            | CCD.META_LLAMA4_MAVERICK_17B_INSTRUCT_V1
            | CCD.META_LLAMA4_SCOUT_17B_INSTRUCT_V1
            | CCD.COHERE_COMMAND_R_V1
            | CCD.COHERE_COMMAND_R_PLUS_V1
            | CCD.DEEPSEEK_R1_V2
            | CCD.MINIMAX_M25
        ):
            return default_converse_tokenizer_factory
        case (
            CCD.ANTHROPIC_CLAUDE_V4_5_HAIKU
            | CCD.ANTHROPIC_CLAUDE_V4_SONNET
            | CCD.ANTHROPIC_CLAUDE_V4_5_SONNET
            | CCD.ANTHROPIC_CLAUDE_V4_6_OPUS
            | CCD.ANTHROPIC_CLAUDE_V4_6_SONNET
            | CCD.ANTHROPIC_CLAUDE_V4_1_OPUS
            | CCD.ANTHROPIC_CLAUDE_V5_FABLE
        ):
            return upstream_converse_tokenizer_factory
        case (
            CCD.STABILITY_STABLE_DIFFUSION_3_5_LARGE_V1
            | CCD.STABILITY_STABLE_IMAGE_CORE_V1_1
            | CCD.STABILITY_STABLE_IMAGE_ULTRA_V1
            | CCD.STABILITY_STABLE_IMAGE_ULTRA_V1_1
        ):
            raise ValueError(
                "Stability AI deployments are not supported by Converse API adapter."
            )
        case _:
            assert_never(deployment)


class ConverseAdapterFactory(BaseModel):
    deployment: AdapterChatCompletionDeployment
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
            deployment=self.deployment.upstream_deployment_id,
            bedrock=await self.get_client(),
            storage=create_file_storage(self.api_key),
            input_tokenizer_factory=_get_tokenizer_factory(
                self.deployment.reference_deployment_id
            ),
            support_tools=tools_support != ToolsSupport.NONE,
            supported_image_types=supported_image_types or [],
            supported_document_types=supported_document_types or [],
            ensure_non_empty_tool_descriptions=ensure_non_empty_tool_descriptions,
        )
        return compose_decorators(
            preprocess_messages_decorator(default_preprocess_messages),
            replicator_decorator(),
            caching_decorator(),
        )(model)
