from typing import assert_never

from aidial_sdk.chat_completion import Request as ChatCompletionRequest

import aidial_adapter_bedrock.llm.model.claude.v3.adapter as claude_v3
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.embedding.amazon.titan_image import (
    AmazonTitanImageEmbeddings,
)
from aidial_adapter_bedrock.embedding.amazon.titan_text import (
    AmazonTitanTextEmbeddings,
)
from aidial_adapter_bedrock.embedding.cohere.embed_text import (
    CohereTextEmbeddings,
)
from aidial_adapter_bedrock.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.converse.configuration import (
    has_converse_api_configuration,
)
from aidial_adapter_bedrock.llm.converse.factory import (
    ConverseAdapterFactory,
    ToolsSupport,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDocumentType,
    ConverseImageType,
)
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.model.stability.v2 import StabilityV2Adapter
from aidial_adapter_bedrock.upstream_config import UpstreamConfig
from aidial_adapter_bedrock.utils.adapter_deployments import (
    AdapterChatCompletionDeployment,
    AdapterEmbeddingsDeployment,
)


async def get_bedrock_adapter(
    *,
    deployment: AdapterChatCompletionDeployment,
    api_key: str,
    upstream_config: UpstreamConfig,
    request: ChatCompletionRequest | None,
) -> ChatCompletionAdapter:
    model = deployment.upstream_deployment_id

    async def get_bedrock_client():
        return await Bedrock.acreate(upstream_config)

    converse_adapter = ConverseAdapterFactory(
        deployment=model, get_client=get_bedrock_client, api_key=api_key
    )

    match deployment.reference_deployment_id:
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_OPUS
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_1_OPUS
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_5_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_5_HAIKU
        ):
            if request and has_converse_api_configuration(request):
                return await converse_adapter.create(
                    tools_support=ToolsSupport.ALWAYS,
                    supported_image_types=ConverseImageType.all(),
                    supported_document_types=ConverseDocumentType.all(),
                )
            else:
                return await claude_v3.create_adapter(
                    deployment.clone(deployment.reference_deployment_id),
                    api_key,
                    upstream_config,
                )
        case (
            ChatCompletionDeployment.AI21_JAMBA_1_5_LARGE_V1
            | ChatCompletionDeployment.AI21_JAMBA_1_5_MINI_V1
        ):
            return await converse_adapter.create(
                tools_support=ToolsSupport.NON_STREAMING_ONLY,
                supported_document_types=ConverseDocumentType.all(),
            )
        case (
            ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1_1
            | ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1
            | ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1_1
            | ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_5_LARGE_V1
        ):
            adapter = StabilityV2Adapter.create(
                await get_bedrock_client(),
                deployment.clone(deployment.reference_deployment_id),
                api_key,
            )
            return replicator_decorator()(adapter)
        case (
            ChatCompletionDeployment.COHERE_COMMAND_R_V1
            | ChatCompletionDeployment.COHERE_COMMAND_R_PLUS_V1
        ):
            return await converse_adapter.create(
                tools_support=ToolsSupport.ALWAYS,
                # Cohere models fail on missing tool description:
                # "validationException: Invalid parameter combination",
                # even though other Converse models are fine with it.
                ensure_non_empty_tool_descriptions=True,
                supported_document_types=ConverseDocumentType.all(),
            )
        case (
            ChatCompletionDeployment.AMAZON_NOVA_MICRO
            | ChatCompletionDeployment.AMAZON_NOVA_PRO
            | ChatCompletionDeployment.AMAZON_NOVA_LITE
        ):
            return await converse_adapter.create(
                tools_support=ToolsSupport.ALWAYS,
                supported_image_types=ConverseImageType.all(),
                supported_document_types=ConverseDocumentType.all(),
            )
        case (
            ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1
        ):
            return await converse_adapter.create(
                supported_image_types=ConverseImageType.all(),
            )
        case (
            ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_2_1B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_2_3B_INSTRUCT_V1
            | ChatCompletionDeployment.DEEPSEEK_R1_V2
        ):
            return await converse_adapter.create()
        case (
            ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_3_70B_INSTRUCT_V1
        ):
            return await converse_adapter.create(
                tools_support=ToolsSupport.NON_STREAMING_ONLY,
            )
        case (
            ChatCompletionDeployment.META_LLAMA3_2_11B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA4_MAVERICK_17B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA4_SCOUT_17B_INSTRUCT_V1
        ):
            return await converse_adapter.create(
                tools_support=ToolsSupport.NON_STREAMING_ONLY,
                supported_image_types=ConverseImageType.all(),
            )
        case _:
            assert_never(deployment.reference_deployment_id)


async def get_embeddings_model(
    *,
    deployment: AdapterEmbeddingsDeployment,
    api_key: str,
    upstream_config: UpstreamConfig,
) -> EmbeddingsAdapter:
    model = deployment.upstream_deployment_id
    client = await Bedrock.acreate(upstream_config)
    match deployment.reference_deployment_id:
        case EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V1:
            return AmazonTitanTextEmbeddings.create(
                client, model, supports_dimensions=False
            )
        case EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2:
            return AmazonTitanTextEmbeddings.create(
                client, model, supports_dimensions=True
            )
        case EmbeddingsDeployment.AMAZON_TITAN_EMBED_IMAGE_V1:
            return AmazonTitanImageEmbeddings.create(client, model, api_key)
        case (
            EmbeddingsDeployment.COHERE_EMBED_ENGLISH_V3
            | EmbeddingsDeployment.COHERE_EMBED_MULTILINGUAL_V3
        ):
            return CohereTextEmbeddings.create(client, model)
        case _:
            assert_never(deployment)
