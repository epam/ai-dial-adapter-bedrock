import json

import pytest
from fastapi import Request

from aidial_adapter_bedrock import anthropic_passthrough
from aidial_adapter_bedrock.anthropic_passthrough import (
    _get_model,
    _resolve_session_tags,
)
from aidial_adapter_bedrock.upstream_config import (
    AWSAssumeRoleCredentials,
    CloudUpstreamConfig,
)


def _request(body: bytes, headers: dict[str, str] | None = None) -> Request:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/anthropic/v1/messages",
            "headers": [
                (key.lower().encode(), value.encode())
                for key, value in (headers or {}).items()
            ],
        }
    )
    # Prime the body cache the way Starlette does once the body is read.
    request._body = body
    return request


def _assume_role_config() -> CloudUpstreamConfig:
    return CloudUpstreamConfig(
        region="us-east-1",
        claude_client="legacy",
        credentials=AWSAssumeRoleCredentials(aws_assume_role_arn="arn"),
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            json.dumps({"model": "anthropic.claude-opus-5"}),
            "anthropic.claude-opus-5",
        ),
        # A malformed or model-less body mustn't sink the other tag sources.
        (json.dumps({"messages": []}), None),
        (json.dumps({"model": 15}), None),
        ("not json at all", None),
        ("", None),
    ],
)
async def test_get_model(body: str, expected: str | None):
    assert await _get_model(_request(body.encode())) == expected


async def test_get_model_reads_the_cached_body_only():
    """The proxy reads the same request afterwards, so the body must survive."""

    body = json.dumps({"model": "anthropic.claude-opus-5"}).encode()
    request = _request(body)

    assert await _get_model(request) == "anthropic.claude-opus-5"
    assert await request.body() == body


async def test_resolve_session_tags_skips_the_body_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        anthropic_passthrough, "session_tags_enabled", lambda config: False
    )

    request = _request(b"")  # Reading it as JSON would throw.

    assert await _resolve_session_tags(request, _assume_role_config()) is None


async def test_resolve_session_tags_passes_the_model_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    async def _fake_resolve(api_key, upstream_config, deployment):
        captured.update(api_key=api_key, deployment=deployment)
        return None

    monkeypatch.setattr(
        anthropic_passthrough, "session_tags_enabled", lambda config: True
    )
    monkeypatch.setattr(
        anthropic_passthrough, "resolve_session_tags", _fake_resolve
    )

    request = _request(
        json.dumps({"model": "anthropic.claude-opus-5"}).encode(),
        {"Api-Key": "dial-key"},
    )

    await _resolve_session_tags(request, _assume_role_config())

    assert captured == {
        "api_key": "dial-key",
        "deployment": "anthropic.claude-opus-5",
    }
