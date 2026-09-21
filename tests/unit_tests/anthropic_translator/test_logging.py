import logging

import pytest

from tests.unit_tests.anthropic_translator.helpers import convert, user


def test_conversion_warnings_use_the_project_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        convert({**user("hi"), "top_k": 2, "inference_geo": "eu"})
    records = [record for record in caplog.records if record.name == "bedrock"]
    assert any(
        record.levelno == logging.DEBUG and "top_k" in record.message
        for record in records
    )
    assert any(
        record.levelno == logging.WARNING and "inference_geo" in record.message
        for record in records
    )
