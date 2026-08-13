from datetime import UTC, datetime, timedelta

import pytest

from aidial_adapter_bedrock.anthropic_translator.chat_completions.cache_breakpoints import (
    cache_breakpoint,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)


def marker(*controls):
    return cache_breakpoint(list(controls), TranslationLog("test"))


def seconds_from_now(expire_at: str | None) -> float:
    assert expire_at is not None
    return (
        datetime.fromisoformat(expire_at) - datetime.now(UTC)
    ).total_seconds()


def test_no_cache_control_means_no_marker():
    assert marker() is None


def test_a_cache_control_without_a_ttl_leaves_the_default_alone():
    # An empty breakpoint leaves Core and the provider on their own default.
    breakpoint = marker({"type": "ephemeral"})
    assert breakpoint is not None
    assert breakpoint.expire_at is None


@pytest.mark.parametrize(
    "ttl, expected",
    [
        ("5m", timedelta(minutes=5)),
        ("1h", timedelta(hours=1)),
        ("30s", timedelta(seconds=30)),
        ("2d", timedelta(days=2)),
        ("0m", timedelta(0)),
    ],
)
def test_a_ttl_duration_becomes_an_absolute_instant(ttl, expected):
    # Anthropic states a lifetime as a duration on the block; DIAL takes the
    # instant it lands on.
    breakpoint = marker({"type": "ephemeral", "ttl": ttl})
    assert breakpoint is not None
    assert breakpoint.expire_at is not None
    assert breakpoint.expire_at.endswith("Z")
    assert seconds_from_now(breakpoint.expire_at) == pytest.approx(
        expected.total_seconds(), abs=2
    )


@pytest.mark.parametrize(
    "ttl", ["", "forever", "5", "m", "5 m", "-1h", 300, []]
)
def test_an_unreadable_ttl_never_costs_the_marker(ttl):
    # Dropping the marker would stop caching altogether; falling back to the
    # upstream's default only loses the requested lifetime.
    breakpoint = marker({"type": "ephemeral", "ttl": ttl})
    assert breakpoint is not None
    assert breakpoint.expire_at is None


def test_the_longest_ttl_wins_when_blocks_collapse_onto_one_object():
    # Taking the maximum can only keep content cached longer than one source
    # asked for; taking the minimum would silently shorten another's.
    breakpoint = marker(
        {"ttl": "5m"}, {"ttl": "1h"}, {"type": "ephemeral"}, {"ttl": "30s"}
    )
    assert breakpoint is not None
    assert seconds_from_now(breakpoint.expire_at) == pytest.approx(3600, abs=2)


def test_an_unreadable_ttl_does_not_shadow_a_readable_one():
    breakpoint = marker({"ttl": "nonsense"}, {"ttl": "1h"})
    assert breakpoint is not None
    assert seconds_from_now(breakpoint.expire_at) == pytest.approx(3600, abs=2)


def test_the_anthropic_cache_control_type_is_never_forwarded():
    # `type` means nothing to DIAL.
    breakpoint = marker({"type": "ephemeral", "ttl": "5m"})
    assert breakpoint is not None
    assert breakpoint.model_dump(exclude_none=True).keys() == {"expire_at"}
