from typing import Callable

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.auth import Auth
from aidial_adapter_bedrock.llm.chat_emulator import default_emulator
from aidial_adapter_bedrock.llm.chat_model import ChatModel, Model
from aidial_adapter_bedrock.llm.model.ai21 import AI21Adapter
from aidial_adapter_bedrock.llm.model.amazon import AmazonAdapter
from aidial_adapter_bedrock.llm.model.anthropic import AnthropicAdapter
from aidial_adapter_bedrock.llm.model.cohere import CohereAdapter
from aidial_adapter_bedrock.llm.model.meta import MetaAdapter
from aidial_adapter_bedrock.llm.model.stability import StabilityAdapter
from aidial_adapter_bedrock.llm.tokenize import default_tokenize


async def get_bedrock_adapter(
    model: str, region: str, get_auth: Callable[[], Auth]
) -> ChatModel:
    client = await Bedrock.acreate(region)
    provider = Model.parse(model).provider
    match provider:
        case "anthropic":
            return AnthropicAdapter(client, model)
        case "ai21":
            return AI21Adapter(
                client, model, default_tokenize, default_emulator
            )
        case "stability":
            return StabilityAdapter(client, model, get_auth)
        case "amazon":
            return AmazonAdapter(
                client, model, default_tokenize, default_emulator
            )
        case "meta":
            return MetaAdapter(client, model, default_tokenize)
        case "cohere":
            return CohereAdapter(client, model, default_tokenize)
        case _:
            raise ValueError(f"Unknown model provider: '{provider}'")
