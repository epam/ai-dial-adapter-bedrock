import pytest
from aidial_sdk.chat_completion.request import ReasoningEffort

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    UNRESOLVED_PROFILE,
    NestedBudget,
    NestedEffort,
    TopLevelEffort,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    resolve_reasoning,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)
from tests.unit_tests.anthropic_translator.helpers import (
    ALL_EFFORTS,
    make_profile,
)


def resolve(body: dict, profile=None):
    request = MessagesRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], **body}
    )
    return resolve_reasoning(
        request,
        profile if profile is not None else make_profile(),
        TranslationLog("test"),
    )


@pytest.mark.parametrize(
    "effort, expected",
    [
        ("low", ReasoningEffort.LOW),
        ("medium", ReasoningEffort.MEDIUM),
        ("high", ReasoningEffort.HIGH),
        # Anthropic's enum is wider than OpenAI's, and an unknown value fails
        # the whole request.
        ("xhigh", ReasoningEffort.HIGH),
        ("max", ReasoningEffort.HIGH),
        # An explicit opt-out is not "low".
        ("none", None),
    ],
)
def test_output_config_effort_maps(effort, expected):
    assert resolve({"output_config": {"effort": effort}}) == (expected, {})


@pytest.mark.parametrize(
    "budget, expected",
    [
        (0, None),
        (-1, None),
        (1, ReasoningEffort.LOW),
        (8000, ReasoningEffort.LOW),
        (8001, ReasoningEffort.MEDIUM),
        (24000, ReasoningEffort.MEDIUM),
        (24001, ReasoningEffort.HIGH),
        (999999, ReasoningEffort.HIGH),
    ],
)
def test_thinking_budget_falls_back_to_the_effort_ladder(budget, expected):
    assert resolve(
        {"thinking": {"type": "enabled", "budget_tokens": budget}}
    ) == (expected, {})


def test_output_config_effort_outranks_the_budget():
    effort, _ = resolve(
        {
            "output_config": {"effort": "low"},
            "thinking": {"type": "enabled", "budget_tokens": 999999},
        }
    )
    assert effort == ReasoningEffort.LOW


def test_an_unknown_effort_falls_through_to_the_budget():
    effort, _ = resolve(
        {
            "output_config": {"effort": "turbo"},
            "thinking": {"type": "enabled", "budget_tokens": 30000},
        }
    )
    assert effort == ReasoningEffort.HIGH


def test_a_boolean_budget_is_not_read_as_a_number():
    # Python's `bool` is an `int` subclass, so `True` must not become 1 — and
    # a malformed value degrades rather than failing the request.
    assert resolve({"thinking": {"budget_tokens": True}}) == (None, {})


def test_no_reasoning_signal_emits_nothing():
    assert resolve({}) == (None, {})


def test_thinking_without_a_budget_emits_nothing():
    assert resolve({"thinking": {"type": "disabled"}}) == (None, {})


@pytest.mark.parametrize(
    "levels, expected",
    [
        # An effort outside the advertised list is clamped to the closest one.
        (["low", "medium"], ReasoningEffort.MEDIUM),
        (["low"], ReasoningEffort.LOW),
        (["minimal"], ReasoningEffort.MINIMAL),
        (ALL_EFFORTS, ReasoningEffort.HIGH),
    ],
)
def test_effort_is_clamped_to_the_advertised_levels(levels, expected):
    effort, _ = resolve(
        {"output_config": {"effort": "high"}},
        make_profile(reasoning=TopLevelEffort(levels=levels)),
    )
    assert effort == expected


def test_an_empty_advertised_list_emits_nothing():
    # An empty list is a real answer meaning "supports no reasoning".
    assert resolve(
        {"output_config": {"effort": "high"}},
        make_profile(reasoning=TopLevelEffort(levels=[])),
    ) == (None, {})


def test_levels_this_dialect_cannot_express_are_ignored():
    assert resolve(
        {"output_config": {"effort": "high"}},
        make_profile(reasoning=TopLevelEffort(levels=["xhigh"])),
    ) == (None, {})


def test_an_unresolved_profile_emits_nothing():
    # Unknown is not unsupported: sending nothing degrades a request, sending
    # the wrong knob fails it.
    assert resolve(
        {"output_config": {"effort": "high"}}, UNRESOLVED_PROFILE
    ) == (None, {})


def test_a_reasoning_deployment_gets_the_effort_nested_and_merged():
    effort, configuration = resolve(
        {"output_config": {"effort": "medium"}},
        make_profile(reasoning=NestedEffort(defaults={"summary": "auto"})),
    )
    assert effort is None
    assert configuration == {
        "reasoning": {"summary": "auto", "effort": "medium"}
    }


def test_a_reasoning_deployment_without_an_effort_emits_nothing():
    assert resolve(
        {}, make_profile(reasoning=NestedEffort(defaults={"summary": "auto"}))
    ) == (None, {})


def test_a_thinking_deployment_receives_the_raw_budget():
    effort, configuration = resolve(
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
        make_profile(
            reasoning=NestedBudget(defaults={"include_thoughts": True})
        ),
    )
    assert effort is None
    assert configuration == {
        "thinking": {"include_thoughts": True, "thinking_budget": 4096}
    }


def test_a_thinking_deployment_never_receives_a_reasoning_effort():
    """★ The request that caused the outage: a deployment configured with a
    thinking budget rejects anything carrying `reasoning_effort`, because the
    Vertex adapter maps it onto Gemini's `thinking_level`."""
    effort, configuration = resolve(
        {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
        make_profile(
            reasoning=NestedBudget(defaults={"thinking_budget": 2048})
        ),
    )
    # The request names no budget, so the deployment's own configured one
    # stands and nothing at all is emitted.
    assert effort is None
    assert configuration == {}


def test_exactly_one_knob_is_ever_emitted():
    for reasoning in (
        NestedEffort(defaults={}),
        NestedBudget(defaults={}),
        TopLevelEffort(levels=ALL_EFFORTS),
    ):
        effort, configuration = resolve(
            {
                "output_config": {"effort": "high"},
                "thinking": {"type": "enabled", "budget_tokens": 30000},
            },
            make_profile(reasoning=reasoning),
        )
        assert (effort is None) or (configuration == {}), reasoning
