from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter


class ConverseAdapterWithStreamingEmulation(ConverseAdapter):
    """
    Llama 3 models support tools only in the non-streaming mode.
    So we need to run request in non-streaming mode, and then emulate streaming.
    """

    def is_stream(self, params: ModelParameters) -> bool:
        if self.get_tool_config(params):
            return False
        return params.stream
