from typing import Mapping, assert_never

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.deployments import BedrockDeployment
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
from aidial_adapter_bedrock.llm.model.meta import MetaAdapter
from aidial_adapter_bedrock.llm.model.stability import StabilityAdapter


async def get_bedrock_adapter(
    deployment: BedrockDeployment, region: str, headers: Mapping[str, str]
) -> ChatCompletionAdapter:
    model = deployment.model_id
    match deployment:
        case (
            BedrockDeployment.ANTHROPIC_CLAUDE_V3_SONNET
            | BedrockDeployment.ANTHROPIC_CLAUDE_V3_HAIKU
            | BedrockDeployment.ANTHROPIC_CLAUDE_V3_OPUS
        ):
            return Claude_V3.create(model, region, headers)
        case (
            BedrockDeployment.ANTHROPIC_CLAUDE_INSTANT_V1
            | BedrockDeployment.ANTHROPIC_CLAUDE_V2
            | BedrockDeployment.ANTHROPIC_CLAUDE_V2_1
        ):
            return await Claude_V1_V2.create(
                await Bedrock.acreate(region), model
            )
        case (
            BedrockDeployment.AI21_J2_JUMBO_INSTRUCT
            | BedrockDeployment.AI21_J2_GRANDE_INSTRUCT
        ):
            return AI21Adapter.create(await Bedrock.acreate(region), model)
        case (
            BedrockDeployment.STABILITY_STABLE_DIFFUSION_XL
            | BedrockDeployment.STABILITY_STABLE_DIFFUSION_XL_V1
        ):
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
        case _:
            assert_never(deployment)
