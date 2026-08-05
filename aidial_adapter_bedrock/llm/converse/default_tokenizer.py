import json
from typing import Any

from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseMessages,
    ConverseRequestWrapper,
)
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string


def default_converse_tokenizer_factory(
    deployment: ConverseDeployment, params: ConverseRequestWrapper
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
