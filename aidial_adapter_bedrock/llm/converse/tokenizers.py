import json
from typing import Any

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseMessages,
    ConverseRequestWrapper,
)
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string


def default_converse_tokenizer_factory(
    deployment: ConverseDeployment,
    bedrock: Bedrock,
    params: ConverseRequestWrapper,
):
    def _to_string(obj: Any) -> str:
        return json.dumps(obj, default=str)

    tool_tokens = default_tokenize_string(_to_string(params.toolConfig))
    system_tokens = default_tokenize_string(_to_string(params.system))

    async def tokenizer(msg_items: ConverseMessages) -> int:
        tokens = sum(
            default_tokenize_string(_to_string(msg_item[0]))
            for msg_item in msg_items
        )
        return tokens + tool_tokens + system_tokens

    return tokenizer


def upstream_converse_tokenizer_factory(
    deployment: ConverseDeployment,
    bedrock: Bedrock,
    params: ConverseRequestWrapper,
):
    async def tokenizer(msg_items: ConverseMessages) -> int:
        # No need to pass `msg_items` since `params.messages` already contain this data
        return await bedrock.acount_tokens(
            model=deployment, **params.to_request()
        )

    return tokenizer
