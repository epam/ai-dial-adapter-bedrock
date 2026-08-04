"""
The one reasoning knob an outbound request may carry.

Anthropic expresses reasoning as a token budget (`thinking.budget_tokens`) or
an effort level (`output_config.effort`); the deployments Core fronts accept
one of three mutually exclusive shapes. Emitting more than one, or the wrong
one, fails the whole request upstream — so exactly one is chosen from the
deployment's capability profile, and none at all when nothing is known.
"""

from typing import Any, Literal, assert_never

from aidial_sdk.chat_completion.request import ReasoningEffort

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
    ThinkingConfig,
)
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    DeploymentProfile,
    NestedBudget,
    NestedEffort,
    TopLevelEffort,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

Effort = Literal["low", "medium", "high"]

# Preference order when the resolved level isn't one the deployment advertises.
_CLAMP_ORDER: tuple[Effort, ...] = ("high", "medium", "low")

_EFFORT_VALUES = frozenset(effort.value for effort in ReasoningEffort)


def resolve_reasoning(
    req: MessagesRequest, profile: DeploymentProfile, tlog: TranslationLog
) -> tuple[ReasoningEffort | None, dict[str, Any]]:
    """The top-level `reasoning_effort` and the `custom_fields.configuration`
    fragment to emit — at most one of the two is ever non-empty."""
    match profile.reasoning:
        case NestedEffort(defaults=defaults):
            effort = _resolve_effort(req, tlog)
            if effort is None:
                return None, {}
            # Merge, don't replace: the deployment's other defaults must stand.
            return None, {"reasoning": {**defaults, "effort": effort}}

        case NestedBudget(defaults=defaults):
            # A budget-based adapter receives the budget verbatim. Bucketing it
            # into an effort here is what the Vertex adapter turns into a
            # `thinking_level`, which its upstream then rejects.
            budget = _budget(req.thinking)
            if budget is None:
                # The deployment's own pre-wired budget stands.
                return None, {}
            return None, {"thinking": {**defaults, "thinking_budget": budget}}

        case TopLevelEffort(levels=levels):
            effort = _resolve_effort(req, tlog)
            supported = [level for level in levels if level in _EFFORT_VALUES]
            if effort is None or not supported:
                return None, {}
            return ReasoningEffort(_clamp(effort, supported, tlog)), {}

        case _:
            assert_never(profile.reasoning)


def _resolve_effort(
    req: MessagesRequest, tlog: TranslationLog
) -> Effort | None:
    """An explicit `output_config.effort` always outranks the budget, whose
    bucketing is lossy in both directions and therefore only a fallback."""
    match req.output_config.effort if req.output_config else None:
        case "low" | "medium" | "high" as effort:
            return effort
        case "xhigh" | "max":
            # Anthropic's enum is wider than OpenAI's, and an unknown value
            # fails the whole request.
            return "high"
        case "none":
            # An explicit opt-out is not "low".
            return None
        case None:
            pass
        case unknown:
            tlog.warning("Unknown output_config.effort: %s", unknown)

    return _from_budget(_budget(req.thinking))


def _budget(thinking: ThinkingConfig | None) -> int | None:
    return thinking.budget_tokens if thinking else None


def _from_budget(budget: int | None) -> Effort | None:
    if budget is None or budget <= 0:
        return None
    if budget <= 8000:
        return "low"
    if budget <= 24000:
        return "medium"
    return "high"


def _clamp(effort: Effort, levels: list[str], tlog: TranslationLog) -> str:
    if effort in levels:
        return effort
    clamped = next(
        (level for level in _CLAMP_ORDER if level in levels), levels[0]
    )
    tlog.debug("Clamping reasoning effort %s to %s", effort, clamped)
    return clamped
