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
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)


def _bedrock_records(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "bedrock" and r.levelno == level
    ]


def test_flush_aggregates_one_line_per_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tlog: TranslationLog = TranslationLog("op")
    tlog.warning("first warning")
    tlog.warning("second %s", "warning")
    tlog.debug("a debug note")

    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        tlog.flush()
        tlog.flush()

    assert _bedrock_records(caplog, logging.WARNING) == [
        "op (2): first warning; second warning"
    ]
    assert _bedrock_records(caplog, logging.DEBUG) == ["op (1): a debug note"]


def test_public_translation_flushes_even_when_it_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    req: MessagesRequest = MessagesRequest.model_validate(
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
        pytest.raises(
            AnthropicHTTPError, match="^Unknown message role: 'nonsense_role'$"
        ),
    ):
        to_chat_completions_request(req, "m", ToolNameAliases())

    assert _bedrock_records(caplog, logging.WARNING) == [
        "Anthropic→Chat Completions request (1): "
        "Dropping unsupported user content block: bogus_block"
    ]
