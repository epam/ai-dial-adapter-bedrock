import random

import pytest

from aidial_adapter_bedrock.anthropic_translator.stop_sequences import (
    StopSequenceMatcher,
    apply_stop_sequences,
    strips_stop_parameter,
)


def stream(sequences: list[str], chunks: list[str]) -> tuple[str, str | None]:
    """Feed `chunks` through the incremental matcher and collect what a client
    would have seen."""
    matcher = StopSequenceMatcher(sequences)
    emitted = "".join(matcher.push(chunk) for chunk in chunks)
    return emitted + matcher.flush(), matcher.matched


@pytest.mark.parametrize(
    "text, sequences, expected_text, expected_match",
    [
        ("hello STOP world", ["STOP"], "hello ", "STOP"),
        ("nothing here", ["STOP"], "nothing here", None),
        ("abc", [], "abc", None),
        # The sequence itself is excluded and nothing after it exists.
        ("keepEND drop", ["END"], "keep", "END"),
        # A sequence at the very start yields empty text.
        ("ENDrest", ["END"], "", "END"),
        # Earliest completion wins, not the earliest start.
        ("abXY", ["abXY", "b"], "a", "b"),
        # Ties on the end offset go to the earlier start.
        ("xxAB", ["AB", "B"], "xx", "AB"),
        # Only the first occurrence matters.
        ("a STOP b STOP c", ["STOP"], "a ", "STOP"),
        ("one-char", ["-"], "one", "-"),
    ],
)
def test_batch_matching(text, sequences, expected_text, expected_match):
    assert apply_stop_sequences(text, sequences) == (
        expected_text,
        expected_match,
    )


@pytest.mark.parametrize(
    "sequences, chunks, expected_text, expected_match",
    [
        (["STOP"], ["hello ", "STOP", " world"], "hello ", "STOP"),
        # Split across chunk boundaries.
        (["STOP"], ["hel", "lo ST", "OP wo"], "hello ", "STOP"),
        (["STOP"], ["S", "T", "O", "P"], "", "STOP"),
        # A partial tail that never completes is released to the client.
        (["STOP"], ["hello ST"], "hello ST", None),
        (["ABCDE"], ["AB"], "AB", None),
        # A single-character sequence withholds nothing.
        (["!"], ["a", "b", "!", "c"], "ab", "!"),
        # No sequences at all passes everything through.
        ([], ["a", "b", "c"], "abc", None),
    ],
)
def test_streaming_matching(sequences, chunks, expected_text, expected_match):
    assert stream(sequences, chunks) == (expected_text, expected_match)


def test_content_after_a_match_is_suppressed():
    matcher = StopSequenceMatcher(["STOP"])
    assert matcher.push("a STOP b") == "a "
    assert matcher.push("more text") == ""
    assert matcher.flush() == ""
    assert matcher.matched == "STOP"


def test_a_single_character_sequence_emits_eagerly():
    # Withhold length is zero here, so nothing may be buffered forever.
    matcher = StopSequenceMatcher(["!"])
    assert matcher.push("abc") == "abc"


@pytest.mark.parametrize("seed", range(200))
def test_batch_and_streaming_agree(seed):
    """The two paths must produce identical text *and* an identical matched
    sequence, across many chunk boundaries — a divergence here is invisible to
    example-based tests."""
    rng = random.Random(seed)  # noqa: S311 — sampling test inputs, not crypto
    alphabet = "abXY"
    text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
    sequences = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
        for _ in range(rng.randint(1, 3))
    ]

    chunks: list[str] = []
    rest = text
    while rest:
        size = rng.randint(1, 3)
        chunks.append(rest[:size])
        rest = rest[size:]

    assert stream(sequences, chunks) == apply_stop_sequences(text, sequences)


@pytest.mark.parametrize(
    "deployment, expected",
    [
        ("gpt-5.5", True),
        ("GPT-5.5", True),
        ("gpt-5.1-mini", True),
        ("gpt-4o", False),
        ("gpt-5", False),  # the default prefix includes the dot
        ("claude-3-5-sonnet", False),
    ],
)
def test_default_stop_unsupported_prefixes(deployment, expected):
    assert strips_stop_parameter(deployment) is expected


def test_stop_unsupported_prefixes_are_configurable(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_STOP_UNSUPPORTED_DEPLOYMENTS", "foo-, Bar-")
    assert strips_stop_parameter("foo-1") is True
    assert strips_stop_parameter("bar-1") is True
    assert strips_stop_parameter("gpt-5.5") is False


def test_an_empty_prefix_list_disables_stripping(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_STOP_UNSUPPORTED_DEPLOYMENTS", "")
    assert strips_stop_parameter("gpt-5.5") is False
