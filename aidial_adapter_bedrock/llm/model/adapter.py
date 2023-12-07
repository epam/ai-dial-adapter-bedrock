from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.chat_emulation.chat_emulator import default_conf
from aidial_adapter_bedrock.llm.chat_model import ChatModel, Model
from aidial_adapter_bedrock.llm.model.ai21 import AI21Adapter
from aidial_adapter_bedrock.llm.model.amazon import AmazonAdapter
from aidial_adapter_bedrock.llm.model.anthropic import AnthropicAdapter
from aidial_adapter_bedrock.llm.model.cohere import CohereAdapter
from aidial_adapter_bedrock.llm.model.meta import MetaAdapter
from aidial_adapter_bedrock.llm.model.stability import StabilityAdapter


def default_tokenize(string: str) -> int:
    """
    The number of bytes is a proxy for the number of tokens for
    models which do not provide any means to count tokens.

    Any token number estimator should satisfy the following requirements:
    1. Overestimation of number of tokens is allowed.
    It's ok to trim the chat history more than necessary.
    2. Underestimation of number of tokens is prohibited.
    It's wrong to leave the chat history as is when the trimming was actually required.
    """
    return len(string.encode("utf-8"))


async def get_bedrock_adapter(model: str, region: str) -> ChatModel:
    client = await Bedrock.acreate(region)
    provider = Model.parse(model).provider
    match provider:
        case "anthropic":
            return AnthropicAdapter(client, model)
        case "ai21":
            return AI21Adapter(client, model, default_tokenize, default_conf)
        case "stability":
            return StabilityAdapter(client, model)
        case "amazon":
            return AmazonAdapter(client, model, default_tokenize, default_conf)
        case "meta":
            return MetaAdapter(client, model, default_tokenize, default_conf)
        case "cohere":
            return CohereAdapter(client, model, default_tokenize)
        case _:
            raise ValueError(f"Unknown model provider: '{provider}'")
