import logging

import pytest

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.to_chat_completions import (
    to_chat_completions_request,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)


def _bedrock_records(caplog, level: int) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "bedrock" and r.levelno == level
    ]


def test_flush_aggregates_one_line_per_level(caplog):
    tlog = TranslationLog("op")
    tlog.warning("first warning")
    tlog.warning("second %s", "warning")
    tlog.debug("a debug note")

    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        tlog.flush()

    # One line per level: operation tag, count, and every recorded message
    # (in recorded order, `%`-formatted) joined into a single record.
    assert _bedrock_records(caplog, logging.WARNING) == [
        "op (2): first warning; second warning"
    ]
    assert _bedrock_records(caplog, logging.DEBUG) == ["op (1): a debug note"]


def test_flush_is_idempotent(caplog):
    tlog = TranslationLog("op")
    tlog.warning("only once")

    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        tlog.flush()
        tlog.flush()  # entries were cleared; a second flush emits nothing

    assert _bedrock_records(caplog, logging.WARNING) == ["op (1): only once"]


def test_flush_without_entries_emits_nothing(caplog):
    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        TranslationLog("op").flush()

    assert [r for r in caplog.records if r.name == "bedrock"] == []


def test_debug_flush_suppressed_below_its_level(caplog):
    tlog = TranslationLog("op")
    tlog.warning("w")
    tlog.debug("d")

    with caplog.at_level(logging.WARNING, logger="bedrock"):
        tlog.flush()

    # The WARNING flush is emitted; the DEBUG flush is skipped by the
    # isEnabledFor guard when the logger sits above DEBUG.
    assert _bedrock_records(caplog, logging.WARNING) == ["op (1): w"]
    assert _bedrock_records(caplog, logging.DEBUG) == []


def test_public_translation_flushes_even_when_it_raises(caplog):
    # A first message records a drop; a later message with an unknown role
    # raises mid-translation. The entry point's `finally` must still flush the
    # note recorded before the raise.
    req = MessagesRequest.model_validate(
        {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": [{"type": "bogus_block"}]},
                {"role": "nonsense_role", "content": "x"},
            ],
        }
    )

    with (
        caplog.at_level(logging.DEBUG, logger="bedrock"),
        pytest.raises(AnthropicHTTPError),
    ):
        to_chat_completions_request(req, "m")

    assert _bedrock_records(caplog, logging.WARNING) == [
        "Anthropic→Chat Completions request (1): "
        "Dropping unsupported user content block: bogus_block"
    ]
