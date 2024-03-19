from typing import Mapping

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from aidial_adapter_bedrock.llm.chat_model import ChatModel, Model
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
    if deployment == BedrockDeployment.ANTHROPIC_CLAUDE_V3:
        return AnthropicChat.create(model, region, headers)

    client = await Bedrock.acreate(region)
    provider = Model.parse(model).provider
    match provider:
        case "anthropic":
            return await AnthropicAdapter.create(client, model)
        case "ai21":
            return AI21Adapter.create(client, model)
        case "stability":
            return StabilityAdapter.create(client, model, headers)
        case "amazon":
            return AmazonAdapter.create(client, model)
        case "meta":
            return MetaAdapter.create(client, model)
        case "cohere":
            return CohereAdapter.create(client, model)
        case _:
            raise ValueError(f"Unknown model provider: '{provider}'")
