import json
from typing import Any, List, Tuple

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseMessage,
    ConverseParams,
)
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string


class ConverseStreamingEmulateAdapter(ConverseAdapter):
    """
    Llama 3.1 supports tools, but only in non-streaming mode.
    So we need to run request in non-streaming mode, and then emulate streaming.
    """

    def is_stream(self, params: ModelParameters) -> bool:
        if self.get_tool_config(params):
            return False
        return params.stream


def input_tokenizer_factory(
    deployment: ConverseDeployment, params: ConverseParams
):
    tool_tokens = default_tokenize_string(json.dumps(params.toolConfig))

    async def tokenizer(msg_items: List[Tuple[ConverseMessage, Any]]) -> int:
        tokens = sum(
            default_tokenize_string(json.dumps(msg_item[0]))
            for msg_item in msg_items
        )
        return tokens + tool_tokens

    return tokenizer
