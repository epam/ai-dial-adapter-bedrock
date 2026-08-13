"""
The `reasoning_effort` an outbound request may carry.

Emitting it unconditionally is not safe: a `gemini-*-with-thinking` deployment
pre-wires a thinking budget in its DIAL config, the Vertex adapter maps
`reasoning_effort` onto Gemini's `thinking_level`, and the pair is rejected
outright. So the request's intent is resolved first, then gated on what the
deployment advertises, then degraded to a level that deployment accepts.
"""

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
    ThinkingConfig,
)
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    DeploymentProfile,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

# Wider than Anthropic's own enum, because what has to be matched is what the
# *deployment* advertises: OpenAI documents none/minimal/low/medium/high/xhigh/
# max and Gemini documents minimal/low/medium/high, both model-dependent.
# `minimum` sits next to `minimal` as a tolerated alias.
EFFORT_LADDER = (
    "none",
    "minimal",
    "minimum",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

# An explicit opt-out. Distinct from `None`, which means the client said
# nothing: only `None` lets a deployment's own default stand, while a
# deployment with pre-wired reasoning has to be *told* to stop.
NO_THINKING = "none"

_LOW_BUDGET = 8000
_MEDIUM_BUDGET = 24000


def resolve_reasoning_effort(
    req: MessagesRequest, profile: DeploymentProfile, tlog: TranslationLog
) -> str | None:
    """The `reasoning_effort` to emit, or nothing at all."""
    advertised = profile.reasoning_efforts
    if not advertised:
        # `[]` is a real answer — "supports no reasoning", and how a deployment
        # with pre-wired thinking keeps the field off its requests. An absent
        # list is silence, and silence is not consent.
        return None

    effort = resolve_effort(req, tlog)
    if effort is None:
        return None
    return degrade_effort(effort, advertised, tlog)


def resolve_effort(req: MessagesRequest, tlog: TranslationLog) -> str | None:
    """What the client asked for, before any deployment gets a say.

    Anthropic's current shape is `thinking: {"type": "adaptive"}` plus
    `output_config.effort`; the deprecated `budget_tokens` is still read as a
    statement of intent.
    """
    thinking: ThinkingConfig | None = req.thinking

    if thinking and thinking.type == "disabled":
        # The opt-out outranks an effort sent alongside it: both fields travel
        # together on real traffic, because `effort` is the *depth* control for
        # thinking, and reading the effort first turns reasoning back on for a
        # request that had just asked for none.
        return NO_THINKING

    match req.output_config.effort if req.output_config else None:
        case None:
            pass
        case effort if effort in EFFORT_LADDER:
            # Verbatim: degradation happens against what the deployment
            # advertises, not against Anthropic's own enum.
            return effort
        case unknown:
            tlog.warning("Unknown output_config.effort: %s", unknown)

    if thinking is None:
        return None
    return _from_budget(thinking.budget_tokens)


def _from_budget(budget: int | None) -> str:
    if budget is None:
        # `adaptive`, `enabled` without a budget, or a type we have never seen.
        # Anthropic documents `high` as the API default and states that
        # omitting the effort "produces identical behavior", so a catch-all
        # here means "think at the default" rather than "don't think".
        return "high"
    if budget <= 0:
        return NO_THINKING
    if budget <= _LOW_BUDGET:
        return "low"
    if budget <= _MEDIUM_BUDGET:
        return "medium"
    return "high"


def degrade_effort(
    effort: str, advertised: list[str], tlog: TranslationLog
) -> str | None:
    """The nearest level on `advertised`, or nothing when there is none.

    An unadvertised level walks **down** first — less thinking is a quality
    loss, more is a cost and latency surprise — then up if there is nothing
    below. `none` is never crossed in either direction: degrading a real effort
    to it silently switches thinking off, and upgrading away from it switches
    thinking back on for a request that asked for none.
    """
    if effort in advertised:
        return effort

    if effort == NO_THINKING:
        tlog.debug("Dropping reasoning_effort: %s is not advertised", effort)
        return None

    index = EFFORT_LADDER.index(effort)
    nearest = [
        *reversed(EFFORT_LADDER[1:index]),
        *EFFORT_LADDER[index + 1 :],
    ]
    degraded = next((level for level in nearest if level in advertised), None)
    if degraded is None:
        tlog.debug("Dropping reasoning_effort: %s is not advertised", effort)
    else:
        tlog.debug("Degrading reasoning effort %s to %s", effort, degraded)
    return degraded
