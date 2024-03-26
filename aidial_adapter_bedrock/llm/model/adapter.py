from typing import Mapping

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.chat_model import ChatModel
from aidial_adapter_bedrock.llm.model.ai21 import AI21Adapter
from aidial_adapter_bedrock.llm.model.amazon import AmazonAdapter
from aidial_adapter_bedrock.llm.model.anthropic import (
    AnthropicAdapter,
    AnthropicChat,
)
from aidial_adapter_bedrock.llm.model.cohere import CohereAdapter
from aidial_adapter_bedrock.llm.model.meta import MetaAdapter
from aidial_adapter_bedrock.llm.model.stability import StabilityAdapter


async def get_bedrock_adapter(
    deployment: BedrockDeployment, region: str, headers: Mapping[str, str]
) -> ChatModel:
    model = deployment.model_id
    match deployment:
        case BedrockDeployment.ANTHROPIC_CLAUDE_V3:
            return AnthropicChat.create(model, region, headers)
        case (
            BedrockDeployment.ANTHROPIC_CLAUDE_INSTANT_V1
            | BedrockDeployment.ANTHROPIC_CLAUDE_V1
            | BedrockDeployment.ANTHROPIC_CLAUDE_V2
            | BedrockDeployment.ANTHROPIC_CLAUDE_V2_1
        ):
            return await AnthropicAdapter.create(
                await Bedrock.acreate(region), model
            )
        case (
            BedrockDeployment.AI21_J2_JUMBO_INSTRUCT
            | BedrockDeployment.AI21_J2_GRANDE_INSTRUCT
        ):
            return AI21Adapter.create(await Bedrock.acreate(region), model)
        case BedrockDeployment.STABILITY_STABLE_DIFFUSION_XL:
            return StabilityAdapter.create(
                await Bedrock.acreate(region), model, headers
            )
        case BedrockDeployment.AMAZON_TITAN_TG1_LARGE:
            return AmazonAdapter.create(await Bedrock.acreate(region), model)
        case (
            BedrockDeployment.META_LLAMA2_13B_CHAT_V1
            | BedrockDeployment.META_LLAMA2_70B_CHAT_V1
        ):
            return MetaAdapter.create(await Bedrock.acreate(region), model)
        case (
            BedrockDeployment.COHERE_COMMAND_TEXT_V14
            | BedrockDeployment.COHERE_COMMAND_LIGHT_TEXT_V14
        ):
            return CohereAdapter.create(await Bedrock.acreate(region), model)
