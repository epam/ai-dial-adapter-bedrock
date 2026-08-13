import pytest

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    UNRESOLVED_PROFILE,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    EFFORT_LADDER,
    resolve_effort,
    resolve_reasoning_effort,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)
from tests.unit_tests.anthropic_translator.helpers import (
    ALL_EFFORTS,
    make_profile,
)


def request(body: dict) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], **body}
    )


def intent(body: dict) -> str | None:
    """Step 1 only: what the client asked for, before any deployment gets a
    say."""
    return resolve_effort(request(body), TranslationLog("test"))


def emitted(body: dict, profile=None) -> str | None:
    return resolve_reasoning_effort(
        request(body),
        profile if profile is not None else make_profile(),
        TranslationLog("test"),
    )


# --- Step 1: the client's intent -------------------------------------------


@pytest.mark.parametrize("effort", EFFORT_LADDER)
def test_a_known_output_config_effort_survives_verbatim(effort):
    # Degradation happens against what the deployment advertises, not against
    # Anthropic's own enum, so nothing is collapsed here.
    assert intent({"output_config": {"effort": effort}}) == effort


def test_an_unknown_effort_falls_through_to_thinking():
    assert (
        intent(
            {
                "output_config": {"effort": "turbo"},
                "thinking": {"type": "enabled", "budget_tokens": 30000},
            }
        )
        == "high"
    )


def test_an_unknown_effort_with_no_thinking_says_nothing():
    assert intent({"output_config": {"effort": "turbo"}}) is None


@pytest.mark.parametrize(
    "thinking, expected",
    [
        # `adaptive` is what Claude Code sends on essentially every request.
        # Anthropic documents `high` as the API default and states that
        # omitting the effort produces identical behaviour.
        ({"type": "adaptive"}, "high"),
        ({"type": "enabled"}, "high"),
        ({"type": "from_2027"}, "high"),
        ({}, "high"),
        # The deprecated budget is still read as a statement of intent.
        ({"type": "enabled", "budget_tokens": 1}, "low"),
        ({"type": "enabled", "budget_tokens": 8000}, "low"),
        ({"type": "enabled", "budget_tokens": 8001}, "medium"),
        ({"type": "enabled", "budget_tokens": 24000}, "medium"),
        ({"type": "enabled", "budget_tokens": 24001}, "high"),
        ({"type": "enabled", "budget_tokens": 999999}, "high"),
        ({"type": "enabled", "budget_tokens": 0}, "none"),
        ({"type": "enabled", "budget_tokens": -1}, "none"),
        # An explicit opt-out.
        ({"type": "disabled"}, "none"),
    ],
)
def test_thinking_states_an_intent(thinking, expected):
    assert intent({"thinking": thinking}) == expected


def test_no_thinking_block_at_all_says_nothing():
    # `None` and `NO_THINKING` are different: only `None` lets a deployment's
    # own default stand.
    assert intent({}) is None


def test_a_boolean_budget_is_not_read_as_a_number():
    # Python's `bool` is an `int` subclass, so `True` must not become 1 — a
    # budget-less `thinking` block means "think at the default".
    assert intent({"thinking": {"budget_tokens": True}}) == "high"


def test_an_opt_out_outranks_an_effort_sent_alongside_it():
    # Both fields travel together on real traffic: a client that switches
    # thinking off keeps sending the effort its config names. Reading the
    # effort first turns reasoning back on for a request that asked for none.
    assert (
        intent(
            {
                "thinking": {"type": "disabled"},
                "output_config": {"effort": "medium"},
            }
        )
        == "none"
    )


def test_an_effort_outranks_a_budget():
    assert (
        intent(
            {
                "output_config": {"effort": "low"},
                "thinking": {"type": "enabled", "budget_tokens": 999999},
            }
        )
        == "low"
    )


# --- Steps 2 and 3: the deployment's gate and the ladder --------------------


@pytest.mark.parametrize(
    "advertised, body, expected",
    [
        # ★ A deployment with a pre-wired thinking budget advertises no
        # efforts, and must never receive `reasoning_effort`: the Vertex
        # adapter maps it onto Gemini's `thinking_level` and the upstream
        # rejects the pair outright.
        ([], {"thinking": {"type": "adaptive"}}, None),
        ([], {"thinking": {"type": "enabled", "budget_tokens": 8000}}, None),
        ([], {"thinking": {"type": "disabled"}}, None),
        ([], {"output_config": {"effort": "high"}}, None),
        # An advertised level is emitted as asked.
        (
            ["low", "high", "xhigh"],
            {"output_config": {"effort": "xhigh"}},
            "xhigh",
        ),
        (ALL_EFFORTS, {"output_config": {"effort": "max"}}, "high"),
        (ALL_EFFORTS, {"thinking": {"type": "adaptive"}}, "high"),
        (
            ALL_EFFORTS,
            {"thinking": {"type": "enabled", "budget_tokens": 24000}},
            "medium",
        ),
        # A positive effort never degrades to `none`.
        (["none", "minimum"], {"output_config": {"effort": "high"}}, "minimum"),
        # `NO_THINKING` is honoured where the deployment can express it.
        (["none", "low"], {"thinking": {"type": "disabled"}}, "none"),
        # ...and dropped where it cannot: sending anything else would switch
        # thinking back on.
        (["low", "high"], {"thinking": {"type": "disabled"}}, None),
        (
            ["none", "low", "medium"],
            {
                "thinking": {"type": "disabled"},
                "output_config": {"effort": "medium"},
            },
            "none",
        ),
        # The client said nothing at all.
        (ALL_EFFORTS, {}, None),
    ],
)
def test_the_worked_examples(advertised, body, expected):
    assert emitted(body, make_profile(reasoning_efforts=advertised)) == expected


@pytest.mark.parametrize(
    "advertised, expected",
    [
        # Down first: less thinking is a quality loss, more is a cost and
        # latency surprise.
        (["low", "medium"], "medium"),
        (["minimal"], "minimal"),
        (["minimum", "low"], "low"),
        # Up only when there is nothing below.
        (["xhigh"], "xhigh"),
        (["xhigh", "max"], "xhigh"),
        (ALL_EFFORTS, "high"),
    ],
)
def test_an_unadvertised_effort_walks_down_then_up(advertised, expected):
    assert (
        emitted(
            {"output_config": {"effort": "high"}},
            make_profile(reasoning_efforts=advertised),
        )
        == expected
    )


def test_an_unresolved_profile_emits_nothing():
    # Unknown is not unsupported: sending nothing degrades a request, sending
    # the wrong knob fails it.
    assert (
        emitted({"output_config": {"effort": "high"}}, UNRESOLVED_PROFILE)
        is None
    )


def test_only_none_is_advertised_so_a_positive_effort_is_dropped():
    assert (
        emitted(
            {"output_config": {"effort": "high"}},
            make_profile(reasoning_efforts=["none"]),
        )
        is None
    )
