from aidial_sdk.chat_completion import FinishReason as DialFinishReason

from aidial_adapter_bedrock.llm.converse.types import ConverseStopReason

CONVERSE_TO_DIAL_FINISH_REASON = {
    ConverseStopReason.END_TURN: DialFinishReason.STOP,
    ConverseStopReason.TOOL_USE: DialFinishReason.TOOL_CALLS,
    ConverseStopReason.MAX_TOKENS: DialFinishReason.LENGTH,
    ConverseStopReason.STOP_SEQUENCE: DialFinishReason.STOP,
    ConverseStopReason.GUARDRAIL_INTERVENED: DialFinishReason.CONTENT_FILTER,
    ConverseStopReason.CONTENT_FILTERED: DialFinishReason.CONTENT_FILTER,
}
