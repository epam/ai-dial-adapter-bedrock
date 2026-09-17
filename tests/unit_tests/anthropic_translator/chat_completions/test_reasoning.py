from typing import cast

import pytest
from anthropic.types.beta.message_create_params import MessageCreateParams

from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    resolve_effort,
)


def intent(body: dict[str, object]):
    return resolve_effort(
        cast(
            MessageCreateParams,
            {"model": "foobar", "max_tokens": 100, "messages": [], **body},
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
