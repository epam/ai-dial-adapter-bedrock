import json

from aidial_adapter_bedrock.llm.converse.adapter import ConverseMessages
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseRequestWrapper,
)
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string


def default_converse_tokenizer_factory(
    deployment: ConverseDeployment, params: ConverseRequestWrapper
):
    tool_tokens = default_tokenize_string(json.dumps(params.toolConfig))
    system_tokens = default_tokenize_string(json.dumps(params.system))

    async def tokenizer(msg_items: ConverseMessages) -> int:
        tokens = sum(
            default_tokenize_string(json.dumps(msg_item[0]))
            for msg_item in msg_items
        )
        return tokens + tool_tokens + system_tokens

    return tokenizer
