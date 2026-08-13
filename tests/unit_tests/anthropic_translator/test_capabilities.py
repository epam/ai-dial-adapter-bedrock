import json

import pytest
from starlette.datastructures import Headers

from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    FEATURES_HEADER,
    UNRESOLVED_PROFILE,
    DeploymentProfile,
    parse_deployment_profile,
)

# A real header captured off a routed request, kept verbatim: it is the only
# evidence of what Core actually publishes, including the flags this translator
# ignores and the `max_completion_tokens_supported: false` every deployment
# seen so far reports.
LIVE_HEADER = json.dumps(
    {
        "rate": True,
        "tokenize": False,
        "truncate_prompt": False,
        "configuration": False,
        "system_prompt": True,
        "tools": False,
        "seed": False,
        "url_attachments": False,
        "folder_attachments": False,
        "allow_resume": True,
        "accessible_by_per_request_key": True,
        "content_parts": True,
        "temperature": True,
        "addons": False,
        "max_completion_tokens_supported": False,
        "reasoning_efforts": ["low", "medium", "high"],
        "cache": True,
    }
)


def parse(raw: str | None) -> DeploymentProfile:
    headers = Headers({FEATURES_HEADER: raw} if raw is not None else {})
    return parse_deployment_profile(headers)


def test_a_live_header_is_read_as_published():
    profile = parse(LIVE_HEADER)
    assert profile == DeploymentProfile(
        temperature_supported=True,
        cache_supported=True,
        max_completion_tokens_supported=False,
        reasoning_efforts=["low", "medium", "high"],
    )


def test_flags_outside_the_four_gates_change_nothing():
    # A header is a declaration, not a proof: this one reports `tools: false`
    # for a deployment that plainly supports tools.
    assert parse(LIVE_HEADER).reasoning_efforts == ["low", "medium", "high"]


@pytest.mark.parametrize(
    "raw",
    [
        None,  # no header at all: the call did not come through Core
        "",
        "not json",
        '"a string"',
        "[]",
        "null",
        "123",
    ],
)
def test_an_unreadable_header_yields_the_unresolved_profile(raw):
    # Unknown is not unsupported, and a capability lookup must never fail the
    # user's message.
    assert parse(raw) == UNRESOLVED_PROFILE


def test_an_empty_object_resolves_to_nothing_advertised():
    # Core answered; this deployment advertises nothing. Behaviourally the same
    # as unresolved — temperature passes, every other gate stays shut.
    assert parse("{}") == UNRESOLVED_PROFILE


def test_the_unresolved_profile_asserts_nothing():
    assert (
        DeploymentProfile(
            # Dropping `temperature` on a guess silently changes generation, so
            # only an explicit `false` suppresses it.
            temperature_supported=True,
            cache_supported=False,
            max_completion_tokens_supported=False,
            reasoning_efforts=[],
        )
        == UNRESOLVED_PROFILE
    )


@pytest.mark.parametrize("value, expected", [(True, True), (False, False)])
def test_temperature_is_dropped_only_on_an_explicit_false(value, expected):
    assert parse(json.dumps({"temperature": value})).temperature_supported is (
        expected
    )


@pytest.mark.parametrize(
    "features, cache, max_completion",
    [
        ({"cache": True}, True, False),
        ({"cache": False}, False, False),
        ({}, False, False),
        ({"max_completion_tokens_supported": True}, False, True),
    ],
)
def test_the_affirmative_gates_need_a_literal_true(
    features, cache, max_completion
):
    profile = parse(json.dumps(features))
    assert profile.cache_supported is cache
    assert profile.max_completion_tokens_supported is max_completion


@pytest.mark.parametrize(
    "features",
    [
        # Core publishes each flag as a JSON boolean; anything else is
        # malformed and must resolve the safe way rather than read as truthy.
        {"cache": "true"},
        {"cache": 1},
        {"temperature": "false"},
        {"reasoning_efforts": "high"},
        {"reasoning_efforts": [1, 2]},
        {"max_completion_tokens_supported": "yes"},
    ],
)
def test_a_non_boolean_flag_is_malformed_not_truthy(features):
    assert parse(json.dumps(features)) == UNRESOLVED_PROFILE


def test_reasoning_efforts_are_carried_through_verbatim():
    # The ladder is wider than Anthropic's own enum because it has to match
    # what the deployment advertises.
    profile = parse(json.dumps({"reasoning_efforts": ["none", "xhigh", "max"]}))
    assert profile.reasoning_efforts == ["none", "xhigh", "max"]


def test_an_empty_reasoning_efforts_list_is_a_real_answer():
    assert parse(json.dumps({"reasoning_efforts": []})).reasoning_efforts == []
