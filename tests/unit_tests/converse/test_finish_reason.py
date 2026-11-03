import pytest
from aidial_sdk.chat_completion import FinishReason as DialFinishReason

from aidial_adapter_bedrock.llm.converse.output import to_dial_finish_reason


@pytest.mark.parametrize(
    "stop_reason, expected",
    [
        ("end_turn", DialFinishReason.STOP),
        ("max_tokens", DialFinishReason.LENGTH),
        ("tool_use", DialFinishReason.TOOL_CALLS),
        ("guardrail_intervened", DialFinishReason.CONTENT_FILTER),
        ("model_context_window_exceeded", DialFinishReason.LENGTH),
    ],
)
def test_to_dial_finish_reason(stop_reason: str, expected: DialFinishReason):
    # Bedrock returns the stop reason as a plain string.
    assert (
        to_dial_finish_reason(stop_reason)  # type: ignore[arg-type]
        == expected
    )
