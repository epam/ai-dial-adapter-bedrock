from aidial_sdk.chat_completion import FinishReason as DialFinishReason
from aidial_sdk.exceptions import RuntimeServerError


def to_dial_finish_reason(converse_stop_reason: str) -> DialFinishReason:
    bedrock_to_dial_finish_reason = {
        "end_turn": DialFinishReason.STOP,
        "tool_use": DialFinishReason.TOOL_CALLS,
        "max_tokens": DialFinishReason.LENGTH,
        "stop_sequence": DialFinishReason.STOP,
        "guardrail_intervened": DialFinishReason.CONTENT_FILTER,
        "content_filtered": DialFinishReason.CONTENT_FILTER,
    }
    if converse_stop_reason not in bedrock_to_dial_finish_reason:
        raise RuntimeServerError(
            f"Unsupported stop reason: {converse_stop_reason}"
        )

    return bedrock_to_dial_finish_reason[converse_stop_reason]
