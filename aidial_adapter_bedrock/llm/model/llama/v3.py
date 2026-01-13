from aidial_adapter_anthropic.dial.request import ModelParameters

from aidial_adapter_bedrock.llm.converse.adapter import ConverseAdapter


class ConverseAdapterWithStreamingEmulation(ConverseAdapter):
    """
    Certain Converse API models support tools only in the non-streaming mode.
    So we need to run request in non-streaming mode, and then emulate streaming.
    https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html
    """

    def is_stream(self, params: ModelParameters) -> bool:
        if params.tool_config is not None:
            return False
        return params.stream
