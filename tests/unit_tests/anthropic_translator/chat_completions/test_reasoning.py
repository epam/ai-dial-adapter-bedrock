import pytest

from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    resolve_effort,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.request import validate_request


def intent(body: dict[str, object]):
    return resolve_effort(
        validate_request(
            {"model": "foobar", "max_tokens": 100, "messages": [], **body}
        )
    )


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_explicit_effort_is_preserved(effort: str) -> None:
    assert (
        intent(
            {
                "output_config": {"effort": effort},
                "thinking": {"type": "adaptive"},
            }
        )
        == effort
    )


@pytest.mark.parametrize(
    "thinking",
    [
        {"type": "adaptive"},
        {"type": "disabled"},
        {"type": "enabled", "budget_tokens": 30000},
    ],
)
def test_invalid_effort_is_rejected(thinking: dict[str, object]) -> None:
    with pytest.raises(AnthropicHTTPError) as exc:
        intent({"output_config": {"effort": "turbo"}, "thinking": thinking})
    assert exc.value.status_code == 400


def test_disabled_thinking_takes_precedence() -> None:
    assert (
        intent(
            {
                "thinking": {"type": "disabled"},
                "output_config": {"effort": "high"},
            }
        )
        == "none"
    )


def test_omission_leaves_deployment_default() -> None:
    assert intent({}) is None


def test_adaptive_thinking_leaves_deployment_default() -> None:
    assert intent({"thinking": {"type": "adaptive"}}) is None


@pytest.mark.parametrize("budget", [1024, 8000, 8001, 24000, 24001, 999999])
def test_budget_does_not_choose_effort(budget: int) -> None:
    assert (
        intent({"thinking": {"type": "enabled", "budget_tokens": budget}})
        is None
    )
    assert (
        intent(
            {
                "thinking": {"type": "enabled", "budget_tokens": budget},
                "output_config": {"effort": "low"},
            }
        )
        == "low"
    )
