from typing import assert_never

import aidial_adapter_bedrock.llm.model.ai21 as ai21
import aidial_adapter_bedrock.llm.model.amazon as amazon
import aidial_adapter_bedrock.llm.model.claude.v1_v2.adapter as claude_v1_v2
import aidial_adapter_bedrock.llm.model.claude.v3.adapter as claude_v3
import aidial_adapter_bedrock.llm.model.cohere as cohere
import aidial_adapter_bedrock.llm.model.stability.v1 as stability_v1
from aidial_adapter_bedrock.adapter_deployments import (
    AdapterChatCompletionDeployment,
    AdapterEmbeddingsDeployment,
)
from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
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


async def get_bedrock_adapter(
    *,
    deployment: AdapterChatCompletionDeployment,
    api_key: str,
    aws_client_config: AWSClientConfig,
) -> ChatCompletionAdapter:
    model = deployment.upstream_deployment_id

    converse_adapter = ConverseAdapterFactory(
        deployment=model, aws_client_config=aws_client_config, api_key=api_key
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
        ):
            return claude_v3.create_adapter(
                deployment.clone(deployment.reference_deployment_id),
                api_key,
                aws_client_config,
            )

        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1
        ):
            return await claude_v1_v2.create_adapter(
                await Bedrock.acreate(aws_client_config), model
            )
        case (
            ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT
            | ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT
            | ChatCompletionDeployment.AI21_J2_MID_V1
            | ChatCompletionDeployment.AI21_J2_ULTRA_V1
        ):
            return ai21.create_adapter(
                await Bedrock.acreate(aws_client_config), model
            )
        case (
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL
            | ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL_V1
        ):
            return stability_v1.create_adapter(
                await Bedrock.acreate(aws_client_config), model, api_key
            )
        case (
            ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1
            | ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1
            | ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1
        ):
            adapter = StabilityV2Adapter.create(
                await Bedrock.acreate(aws_client_config),
                deployment.clone(deployment.reference_deployment_id),
                api_key,
            )
            return replicator_decorator()(adapter)
        case ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE:
            return amazon.create_adapter(
                await Bedrock.acreate(aws_client_config), model
            )
        case (
            ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14
            | ChatCompletionDeployment.COHERE_COMMAND_LIGHT_TEXT_V14
        ):
            return cohere.create_adapter(
                await Bedrock.acreate(aws_client_config), model
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
            | ChatCompletionDeployment.DEEPSEEK_R1_V2_US
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
    aws_client_config: AWSClientConfig,
) -> EmbeddingsAdapter:
    model = deployment.upstream_deployment_id
    client = await Bedrock.acreate(aws_client_config)
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
