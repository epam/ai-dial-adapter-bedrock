import pytest

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    OpenAIEffort,
    resolve_effort,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicHTTPError,
)


def intent(body: dict[str, object]) -> OpenAIEffort | None:
    return resolve_effort(
        MessagesRequest.model_validate({"messages": [], **body})
    )


@pytest.mark.parametrize(
    "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
)
def test_explicit_effort_is_preserved(effort: str) -> None:
    assert (
        intent(
            {
                "output_config": {"effort": effort},
                "thinking": {"type": "adaptive", "budget_tokens": 1},
            }
        )
        == effort
    )


@pytest.mark.parametrize(
    "thinking",
    [None, {"type": "adaptive"}, {"type": "enabled", "budget_tokens": 30000}],
)
def test_invalid_effort_is_rejected(thinking: dict[str, object] | None) -> None:
    with pytest.raises(
        AnthropicHTTPError, match="^Unsupported output_config.effort: turbo$"
    ) as exc:
        intent({"output_config": {"effort": "turbo"}, "thinking": thinking})
    assert exc.value.status_code == 400


def test_disabled_short_circuits_effort_validation() -> None:
    assert (
        intent(
            {
                "thinking": {"type": "disabled"},
                "output_config": {"effort": "turbo"},
            }
        )
        == "none"
    )


def test_omission_leaves_deployment_default() -> None:
    assert intent({}) is None


def test_effort_outranks_budget() -> None:
    assert (
        intent(
            {
                "output_config": {"effort": "low"},
                "thinking": {"budget_tokens": 999999},
            }
        )
        == "low"
    )


@pytest.mark.parametrize(
    "thinking, expected",
    [
        ({"type": "adaptive"}, "high"),
        ({"type": "adaptive", "budget_tokens": 1024}, "high"),
        ({"type": "enabled"}, "high"),
        ({"type": "from_2027"}, "high"),
        ({}, "high"),
        ({"type": "enabled", "budget_tokens": 1}, "low"),
        ({"type": "enabled", "budget_tokens": 8000}, "low"),
        ({"type": "enabled", "budget_tokens": 8001}, "medium"),
        ({"type": "enabled", "budget_tokens": 24000}, "medium"),
        ({"type": "enabled", "budget_tokens": 24001}, "high"),
        ({"type": "enabled", "budget_tokens": 999999}, "high"),
        ({"type": "enabled", "budget_tokens": 0}, "none"),
        ({"type": "enabled", "budget_tokens": -1}, "none"),
        ({"type": "disabled"}, "none"),
    ],
)
def test_thinking_states_an_intent(
    thinking: dict[str, object] | None, expected: str
) -> None:
    assert intent({"thinking": thinking}) == expected
