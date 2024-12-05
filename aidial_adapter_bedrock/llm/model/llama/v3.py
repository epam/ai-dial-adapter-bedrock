import json
from typing import Awaitable, Callable

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.converse.adapter import (
    ConverseAdapter,
    ConverseMessages,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseDeployment,
    ConverseRequestWrapper,
)
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string


class ConverseAdapterWithStreamingEmulation(ConverseAdapter):
    """
    Llama 3 models support tools only in the non-streaming mode.
    So we need to run request in non-streaming mode, and then emulate streaming.
    """

    def is_stream(self, params: ModelParameters) -> bool:
        if self.get_tool_config(params):
            return False
        return params.stream


def input_tokenizer_factory(
    deployment: ConverseDeployment, params: ConverseRequestWrapper
) -> Callable[[ConverseMessages], Awaitable[int]]:
    tool_tokens = default_tokenize_string(json.dumps(params.toolConfig))
    system_tokens = default_tokenize_string(json.dumps(params.system))

    async def tokenizer(msg_items: ConverseMessages) -> int:
        tokens = sum(
            default_tokenize_string(json.dumps(msg_item[0]))
            for msg_item in msg_items
        )
        return tokens + tool_tokens + system_tokens

    return tokenizer
