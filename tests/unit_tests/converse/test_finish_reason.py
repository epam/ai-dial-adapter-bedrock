import pytest
from aidial_sdk.chat_completion import FinishReason as DialFinishReason
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_bedrock.llm.converse.output import to_dial_finish_reason


@pytest.mark.parametrize(
    "stop_reason, expected",
    [
        ("end_turn", DialFinishReason.STOP),
        ("max_tokens", DialFinishReason.LENGTH),
        ("tool_use", DialFinishReason.TOOL_CALLS),
        ("guardrail_intervened", DialFinishReason.CONTENT_FILTER),
    ],
)
def test_to_dial_finish_reason(stop_reason: str, expected: DialFinishReason):
    # Bedrock returns the stop reason as a plain string.
    assert (
        to_dial_finish_reason(stop_reason)  # type: ignore[arg-type]
        == expected
    )


def test_context_window_exceeded_raises_invalid_request():
    with pytest.raises(InvalidRequestError) as exc_info:
        to_dial_finish_reason("model_context_window_exceeded")  # type: ignore[arg-type]

    error = exc_info.value
    assert error.status_code == 400
    assert error.type == "invalid_request_error"
    assert error.code == "context_length_exceeded"
    assert error.param == "messages"
