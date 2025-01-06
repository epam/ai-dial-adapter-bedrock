from typing import assert_never

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
from aidial_adapter_bedrock.dial_api.storage import create_file_storage
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
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.default_tokenizer import (
    default_converse_tokenizer_factory,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDocumentType,
    ConverseImageType,
)
from aidial_adapter_bedrock.llm.model.ai21 import AI21Adapter
from aidial_adapter_bedrock.llm.model.amazon import AmazonAdapter
from aidial_adapter_bedrock.llm.model.claude.v1_v2.adapter import (
    Adapter as Claude_V1_V2,
)
from aidial_adapter_bedrock.llm.model.claude.v3.adapter import (
    Adapter as Claude_V3,
)
from aidial_adapter_bedrock.llm.model.cohere import CohereAdapter
from aidial_adapter_bedrock.llm.model.llama.v3 import (
    ConverseAdapterWithStreamingEmulation,
)
from aidial_adapter_bedrock.llm.model.stability.v1 import StabilityV1Adapter
from aidial_adapter_bedrock.llm.model.stability.v2 import StabilityV2Adapter


async def get_bedrock_adapter(
    *,
    deployment: AdapterChatCompletionDeployment,
    api_key: str,
    aws_client_config: AWSClientConfig,
) -> ChatCompletionAdapter:
    model = deployment.upstream_deployment_id
    match deployment.reference_deployment_id:
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_EU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_EU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US
        ):
            return Claude_V3.create(
                deployment.clone(deployment.reference_deployment_id),
                api_key,
                aws_client_config,
            )
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1
        ):
            return await Claude_V1_V2.create(
                await Bedrock.acreate(aws_client_config), model
            )
        case (
            ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT
            | ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT
            | ChatCompletionDeployment.AI21_J2_MID_V1
            | ChatCompletionDeployment.AI21_J2_ULTRA_V1
        ):
            return AI21Adapter.create(
                await Bedrock.acreate(aws_client_config), model
            )
        case (
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL
            | ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL_V1
        ):
            return StabilityV1Adapter.create(
                await Bedrock.acreate(aws_client_config), model, api_key
            )
        case (
            ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1
            | ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1
        ):
            return StabilityV2Adapter.create(
                await Bedrock.acreate(aws_client_config),
                model,
                api_key,
                image_to_image_supported=False,
            )
        case ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1:
            return StabilityV2Adapter.create(
                await Bedrock.acreate(aws_client_config),
                model,
                api_key,
                image_to_image_supported=True,
                image_width_constraints=(640, 1536),
                image_height_constraints=(640, 1536),
            )
        case ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE:
            return AmazonAdapter.create(
                await Bedrock.acreate(aws_client_config), model
            )
        case (
            ChatCompletionDeployment.AMAZON_NOVA_MICRO
            | ChatCompletionDeployment.AMAZON_NOVA_PRO
            | ChatCompletionDeployment.AMAZON_NOVA_LITE
        ):
            return ConverseAdapter(
                deployment=model,
                bedrock=await Bedrock.acreate(aws_client_config),
                storage=create_file_storage(api_key),
                input_tokenizer_factory=default_converse_tokenizer_factory,
                support_tools=True,
                supported_image_types=ConverseImageType.all(),
                supported_document_types=ConverseDocumentType.all(),
            )
        case (
            ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1
        ):
            return ConverseAdapter(
                deployment=model,
                bedrock=await Bedrock.acreate(aws_client_config),
                storage=create_file_storage(api_key),
                input_tokenizer_factory=default_converse_tokenizer_factory,
                support_tools=False,
                supported_image_types=ConverseImageType.all(),
                supported_document_types=[],
            )
        case (
            ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_2_1B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_2_3B_INSTRUCT_V1
        ):
            return ConverseAdapter(
                deployment=model,
                bedrock=await Bedrock.acreate(aws_client_config),
                storage=create_file_storage(api_key),
                input_tokenizer_factory=default_converse_tokenizer_factory,
                support_tools=False,
                supported_image_types=[],
                supported_document_types=[],
            )
        case (
            ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_2_11B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1
        ):
            is_vision_model = deployment.reference_deployment_id in [
                ChatCompletionDeployment.META_LLAMA3_2_11B_INSTRUCT_V1,
                ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1,
            ]
            return ConverseAdapterWithStreamingEmulation(
                deployment=model,
                bedrock=await Bedrock.acreate(aws_client_config),
                storage=create_file_storage(api_key),
                input_tokenizer_factory=default_converse_tokenizer_factory,
                support_tools=True,
                supported_image_types=(
                    ConverseImageType.all() if is_vision_model else []
                ),
                supported_document_types=[],
            )
        case (
            ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14
            | ChatCompletionDeployment.COHERE_COMMAND_LIGHT_TEXT_V14
        ):
            return CohereAdapter.create(
                await Bedrock.acreate(aws_client_config), model
            )
        case _:
            assert_never(deployment)


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
