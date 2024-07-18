from typing import Mapping, assert_never

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
from aidial_adapter_bedrock.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.model.ai21 import AI21Adapter
from aidial_adapter_bedrock.llm.model.amazon import AmazonAdapter
from aidial_adapter_bedrock.llm.model.claude.v1_v2.adapter import (
    Adapter as Claude_V1_V2,
)
from aidial_adapter_bedrock.llm.model.claude.v3.adapter import (
    Adapter as Claude_V3,
)
from aidial_adapter_bedrock.llm.model.cohere import CohereAdapter
from aidial_adapter_bedrock.llm.model.llama.v2 import llama2_config
from aidial_adapter_bedrock.llm.model.llama.v3 import llama3_config
from aidial_adapter_bedrock.llm.model.meta import MetaAdapter
from aidial_adapter_bedrock.llm.model.stability import StabilityAdapter


async def get_bedrock_adapter(
    deployment: ChatCompletionDeployment,
    region: str,
    headers: Mapping[str, str],
) -> ChatCompletionAdapter:
    model = deployment.model_id
    match deployment:
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS
        ):
            return Claude_V3.create(model, region, headers)
        case (
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2
            | ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1
        ):
            return await Claude_V1_V2.create(
                await Bedrock.acreate(region), model
            )
        case (
            ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT
            | ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT
        ):
            return AI21Adapter.create(await Bedrock.acreate(region), model)
        case (
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL
            | ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL_V1
        ):
            return StabilityAdapter.create(
                await Bedrock.acreate(region), model, headers
            )
        case ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE:
            return AmazonAdapter.create(await Bedrock.acreate(region), model)
        case (
            ChatCompletionDeployment.META_LLAMA2_13B_CHAT_V1
            | ChatCompletionDeployment.META_LLAMA2_70B_CHAT_V1
        ):
            return MetaAdapter.create(
                await Bedrock.acreate(region), model, llama2_config
            )
        case (
            ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1
            | ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1
        ):
            return MetaAdapter.create(
                await Bedrock.acreate(region), model, llama3_config
            )
        case (
            ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14
            | ChatCompletionDeployment.COHERE_COMMAND_LIGHT_TEXT_V14
        ):
            return CohereAdapter.create(await Bedrock.acreate(region), model)
        case _:
            assert_never(deployment)


async def get_embeddings_model(
    deployment: EmbeddingsDeployment,
    region: str,
    headers: Mapping[str, str],
) -> EmbeddingsAdapter:
    model = deployment.model_id
    match deployment:
        case EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V1:
            return AmazonTitanTextEmbeddings.create(
                await Bedrock.acreate(region), model, supports_dimensions=False
            )
        case EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2:
            return AmazonTitanTextEmbeddings.create(
                await Bedrock.acreate(region), model, supports_dimensions=True
            )
        case EmbeddingsDeployment.AMAZON_TITAN_EMBED_IMAGE_V1:
            return AmazonTitanImageEmbeddings.create(
                await Bedrock.acreate(region), model, headers
            )
        case _:
            assert_never(deployment)
