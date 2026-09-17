import json
import logging
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from httpx import ASGITransport

from tests.unit_tests.anthropic_translator.helpers import (
    features_header,
    parse_anthropic_sse,
)

_MESSAGES_URL: str = "/to-chat-completions/anthropic/v1/messages"
_CORE: str = "http://dial-core"


_CORE_PATH: str = "/openai/deployments/gpt-5.5/chat/completions"

_RESPONSE_OBJECT: dict[str, object] = {
    "id": "chatcmpl_1",
    "object": "chat.completion",
    "model": "gpt-5.5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}

_MESSAGES_BODY: dict[str, object] = {
    "model": "gpt-5.5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.fixture
async def client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setenv("DIAL_URL", _CORE)
    from aidial_adapter_bedrock.app import app

    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=ASGITransport(app),
            base_url="http://test-app.com",
            headers={"api-key": "dummy-key"},
        ) as c,
    ):
        yield c


@pytest.fixture
def mock_core() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=_CORE, assert_all_called=False) as mock:
        yield mock


async def test_non_streaming_happy_path(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL, json=_MESSAGES_BODY
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "hi there"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 7


async def test_request_shape_and_headers_not_leaked(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    await client.post(
        _MESSAGES_URL,
        json=_MESSAGES_BODY,
        headers={
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
        },
    )
    sent: httpx.Request = route.calls.last.request
    sent_body = json.loads(sent.content)
    assert sent_body["model"] == "gpt-5.5"

    assert sent_body["max_completion_tokens"] == 100
    assert "max_tokens" not in sent_body

    assert "store" not in sent_body

    assert "anthropic-version" not in sent.headers
    assert "anthropic-beta" not in sent.headers
    assert sent.headers["api-key"] == "dummy-key"
    assert "authorization" not in sent.headers


async def test_attribution_headers_are_forwarded(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    await client.post(
        _MESSAGES_URL,
        json=_MESSAGES_BODY,
        headers={
            "traceparent": "00-trace-span-01",
            "tracestate": "vendor=value",
            "x-claude-code-session-id": "session-1",
            "user-agent": "claude-cli/2.0.0",
        },
    )
    sent: httpx.Request = route.calls.last.request
    assert sent.headers["traceparent"] == "00-trace-span-01"
    assert sent.headers["tracestate"] == "vendor=value"
    assert sent.headers["x-claude-code-session-id"] == "session-1"

    assert (
        sent.headers["user-agent"]
        == "anthropicMessages-to-openaiChatCompletions-translator"
    )


async def test_cores_own_routing_headers_do_not_travel_on(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    await client.post(
        _MESSAGES_URL,
        json=_MESSAGES_BODY,
        headers={
            "x-upstream-endpoint": "https://provider.example/v1",
            "x-upstream-key": "provider-secret",
            "x-dial-override-name": "override",
            **features_header(),
        },
    )
    sent: httpx.Request = route.calls.last.request
    for name in (
        "x-upstream-endpoint",
        "x-upstream-key",
        "x-dial-override-name",
        "x-dial-deployment-features",
    ):
        assert name not in sent.headers


@pytest.mark.parametrize(
    "inbound, forwarded",
    [
        ({"api-key": "core-key"}, "core-key"),
        ({"x-api-key": "client-key"}, None),
        ({"authorization": "Bearer jwt"}, None),
        ({"api-key": "core-key", "x-api-key": "client-key"}, "core-key"),
        ({"api-key": "core-key", "authorization": "Bearer jwt"}, "core-key"),
    ],
)
async def test_only_the_api_key_core_minted_travels_on(
    mock_core: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    inbound: dict[str, str],
    forwarded: str | None,
) -> None:
    monkeypatch.setenv("DIAL_URL", _CORE)
    from aidial_adapter_bedrock.app import app

    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )

    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=ASGITransport(app),
            base_url="http://test-app.com",
        ) as bare,
    ):
        await bare.post(_MESSAGES_URL, json=_MESSAGES_BODY, headers=inbound)

    sent: httpx.Request = route.calls.last.request
    assert sent.headers.get("api-key") == forwarded
    assert "x-api-key" not in sent.headers

    assert "authorization" not in sent.headers


@pytest.mark.parametrize(
    "headers",
    [
        {},
        features_header(
            temperature=False,
            reasoning_efforts=["low"],
            max_completion_tokens_supported=False,
        ),
        features_header(reasoning_efforts=[]),
        {"x-dial-deployment-features": "not json"},
        {"x-dial-deployment-features": "[]"},
    ],
    ids=[
        "absent",
        "unsupported-options",
        "empty-efforts",
        "invalid-json",
        "wrong-shape",
    ],
)
async def test_capabilities_do_not_filter_request_options(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    headers: dict[str, str],
) -> None:
    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={
            **_MESSAGES_BODY,
            "temperature": 0.5,
            "output_config": {"effort": "high"},
        },
        headers=headers,
    )
    assert response.status_code == 200
    assert len(mock_core.calls) == route.call_count == 1
    assert json.loads(route.calls.last.request.content) == {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 100,
        "temperature": 0.5,
        "reasoning_effort": "high",
        "stream": False,
    }


async def test_a_long_mcp_tool_name_round_trips(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    long_name: str = "mcp__" + "s" * 60 + "__do_the_thing"
    route: respx.Route = mock_core.post(_CORE_PATH).mock(
        side_effect=lambda request: httpx.Response(
            200,
            json={
                **_RESPONSE_OBJECT,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": json.loads(request.content)[
                                            "tools"
                                        ][0]["function"]["name"],
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={**_MESSAGES_BODY, "tools": [{"name": long_name}]},
    )

    assert response.status_code == 200
    sent_alias = json.loads(route.calls.last.request.content)["tools"][0][
        "function"
    ]["name"]
    assert sent_alias != long_name
    assert len(sent_alias) <= 64

    assert response.json()["content"][0]["name"] == long_name


async def test_a_long_mcp_tool_name_round_trips_while_streaming(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    long_name: str = "mcp__" + "s" * 60 + "__do_the_thing"

    def echo_the_alias(request: httpx.Request) -> httpx.Response:
        alias = json.loads(request.content)["tools"][0]["function"]["name"]
        frames = [
            {
                "id": "chatcmpl_1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": alias,
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        body: bytes = b"".join(
            b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames
        )
        return httpx.Response(
            200,
            content=body + b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    route: respx.Route = mock_core.post(_CORE_PATH).mock(
        side_effect=echo_the_alias
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={**_MESSAGES_BODY, "stream": True, "tools": [{"name": long_name}]},
    )

    assert response.status_code == 200
    sent_alias = json.loads(route.calls.last.request.content)["tools"][0][
        "function"
    ]["name"]
    assert sent_alias != long_name
    assert len(sent_alias) <= 64

    assert f'"name":"{long_name}"' in response.text
    assert sent_alias not in response.text


async def test_streaming_happy_path(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    upstream_sse: bytes = (
        b'data: {"id": "chatcmpl_1", "model": "gpt-5.5", "choices": '
        b'[{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, '
        b'"finish_reason": null}]}\n\n'
        b'data: {"id": "chatcmpl_1", "choices": [{"index": 0, "delta": {}, '
        b'"finish_reason": "stop"}]}\n\n'
        b'data: {"id": "chatcmpl_1", "choices": [], "usage": '
        b'{"prompt_tokens": 5, "completion_tokens": 2}}\n\n'
        b"data: [DONE]\n\n"
    )
    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        content=upstream_sse, content_type="text/event-stream"
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "stream": True}
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["stream"] is True

    assert sent_body["stream_options"] == {"include_usage": True}
    text: str = response.text
    assert "event: message_start" in text
    assert "event: ping" in text

    assert '"text":"Hello"' in text
    assert "event: message_stop" in text


async def test_x_dial_deployment_id_header_overrides_body_model(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    route: respx.Route = mock_core.post(
        "/openai/deployments/actual-deployment/chat/completions"
    ).respond(json=_RESPONSE_OBJECT, content_type="application/json")

    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json=_MESSAGES_BODY,
        headers={"x-dial-deployment-id": "actual-deployment"},
    )

    assert response.status_code == 200
    assert route.called
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["model"] == "actual-deployment"


async def test_missing_model_returns_400(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == "'model' is required"


async def test_connection_error_to_core_returns_502(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    mock_core.post(_CORE_PATH).mock(side_effect=httpx.ConnectError("refused"))
    response: httpx.Response = await client.post(
        _MESSAGES_URL, json=_MESSAGES_BODY
    )
    assert response.status_code == 502
    body = response.json()
    assert body == {
        "type": "error",
        "error": {"type": "api_error", "message": "Failed to reach DIAL Core"},
    }


async def test_debug_logging_emits_request_and_response_lines(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        response: httpx.Response = await client.post(
            _MESSAGES_URL, json=_MESSAGES_BODY
        )

    assert response.status_code == 200
    messages: list[str] = [
        r.getMessage() for r in caplog.records if r.name == "bedrock"
    ]
    assert any(m.startswith("request: ") for m in messages)
    assert any(m.startswith("response: ") for m in messages)


async def test_debug_logging_emits_stream_chunk_lines(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream_sse: bytes = (
        b'data: {"id": "chatcmpl_1", "model": "gpt-5.5", "choices": '
        b'[{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, '
        b'"finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    mock_core.post(_CORE_PATH).respond(
        content=upstream_sse, content_type="text/event-stream"
    )
    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        response: httpx.Response = await client.post(
            _MESSAGES_URL, json={**_MESSAGES_BODY, "stream": True}
        )

    assert response.status_code == 200
    messages: list[str] = [
        r.getMessage() for r in caplog.records if r.name == "bedrock"
    ]
    assert any(m.startswith("response chunk: ") for m in messages)


async def test_missing_dial_url_returns_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DIAL_URL", raising=False)
    from aidial_adapter_bedrock.app import app

    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=ASGITransport(app),
            base_url="http://test-app.com",
        ) as c,
    ):
        response: httpx.Response = await c.post(
            _MESSAGES_URL, json=_MESSAGES_BODY
        )

    assert response.status_code == 500
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert body["error"]["message"] == "Internal server error"


@pytest.mark.parametrize(
    "status, expected_type",
    [
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (529, "overloaded_error"),
        (418, "api_error"),
    ],
)
async def test_upstream_error_status_mapping(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    status: int,
    expected_type: str,
) -> None:
    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        status_code=status,
        json={"error": {"message": "upstream says no"}},
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL, json=_MESSAGES_BODY
    )

    assert response.status_code == status
    body = response.json()
    assert body["error"]["type"] == expected_type
    assert body["error"]["message"] == "upstream says no"

    assert route.call_count == 1


async def test_pre_stream_error_returns_json_not_sse(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    mock_core.post(_CORE_PATH).respond(
        status_code=401, json={"error": {"message": "bad key"}}
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "stream": True}
    )
    assert response.status_code == 401
    assert "application/json" in response.headers["content-type"]
    assert response.json()["error"]["type"] == "authentication_error"


async def test_malformed_json_returns_400(client: httpx.AsyncClient) -> None:
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_non_object_body_returns_400(client: httpx.AsyncClient) -> None:
    response: httpx.Response = await client.post(_MESSAGES_URL, json=[1, 2, 3])
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_schema_violation_returns_400(client: httpx.AsyncClient) -> None:
    response: httpx.Response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "messages": "not-a-list"}
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == "messages: Input should be a valid list"


@pytest.mark.parametrize("sequences", [["STOP"], ["a", "b", "c", "d", "e"]])
async def test_stop_sequences_are_forwarded(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    sequences: list[str],
) -> None:
    route: respx.Route = mock_core.post(
        "/openai/deployments/gpt-4o/chat/completions"
    ).respond(json=_RESPONSE_OBJECT)
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={**_MESSAGES_BODY, "model": "gpt-4o", "stop_sequences": sequences},
    )
    assert response.status_code == 200
    assert json.loads(route.calls.last.request.content)["stop"] == sequences


async def test_missing_max_tokens_returns_400(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
) -> None:
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "path", ["/v1/messages/count_tokens", "/v1/messages/batches"]
)
async def test_unsupported_endpoint_returns_anthropic_404(
    client: httpx.AsyncClient, path: str
) -> None:
    response: httpx.Response = await client.post(
        "/to-chat-completions/anthropic" + path, json={}
    )
    assert response.status_code == 404
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "not_found_error",
            "message": f"Unknown endpoint: /to-chat-completions/anthropic{path}",
        },
    }


@pytest.mark.parametrize("streaming", [False, True])
async def test_citation_configuration_is_sent_to_core(
    client: httpx.AsyncClient, mock_core: respx.MockRouter, streaming: bool
) -> None:
    request: dict[str, object] = {
        **_MESSAGES_BODY,
        "stream": streaming,
        "unused": "raw input",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "text", "data": "doc"},
                        "citations": {"enabled": True},
                    }
                ],
            }
        ],
    }
    route: respx.Route = mock_core.post(_CORE_PATH)
    if streaming:
        route.respond(
            content=b"data: [DONE]\n\n", content_type="text/event-stream"
        )
    else:
        route.respond(json=_RESPONSE_OBJECT)
    response: httpx.Response = await client.post(_MESSAGES_URL, json=request)
    assert response.status_code == 200
    translated = json.loads(route.calls.last.request.content)
    assert translated["custom_fields"] == {
        "configuration": {"enable_citations": True}
    }
    assert translated["stream"] == streaming
    assert "unused" not in translated


async def test_invalid_effort_returns_anthropic_400(
    client: httpx.AsyncClient,
) -> None:
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={**_MESSAGES_BODY, "output_config": {"effort": "turbo"}},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_cache_policy_and_unlisted_headers_pass_through(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIAL_API_VERSION", "custom-version")
    route: respx.Route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT
    )
    await client.post(
        _MESSAGES_URL,
        json=_MESSAGES_BODY,
        headers={
            "x-dial-cache-policy": "unknown-policy",
            "x-dial-extra": "value",
            "x-dial-cache-breakpoint-path": "inbound-path",
            "x-custom-header": "custom",
        },
    )
    sent: httpx.Request = route.calls.last.request
    assert sent.url.params["api-version"] == "custom-version"
    assert sent.headers["x-dial-cache-policy"] == "unknown-policy"
    assert sent.headers["x-dial-extra"] == "value"
    assert sent.headers["x-custom-header"] == "custom"
    assert "x-dial-cache-breakpoint-path" not in sent.headers


@pytest.mark.parametrize(
    "deployment, emulated",
    [
        ("GPT-5.5", True),
        ("gpt-5.5", True),
        ("gpt-4o", False),
    ],
)
async def test_stop_emulation_uses_resolved_deployment(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    deployment: str,
    emulated: bool,
) -> None:
    route: respx.Route = mock_core.post(
        f"/openai/deployments/{deployment}/chat/completions"
    ).respond(
        json={
            **_RESPONSE_OBJECT,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "before STOP after",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={**_MESSAGES_BODY, "stop_sequences": ["STOP"]},
        headers={"x-dial-deployment-id": deployment},
    )
    assert response.status_code == 200
    sent = json.loads(route.calls.last.request.content)
    assert ("stop" not in sent) == emulated
    assert response.json()["stop_sequence"] == ("STOP" if emulated else None)
    assert response.json()["content"][0]["text"] == (
        "before " if emulated else "before STOP after"
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "usage, cache_write",
    [
        ({}, 0),
        ({"prompt_tokens": None, "completion_tokens": None}, 0),
        (
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "prompt_tokens_details": {
                    "cached_tokens": None,
                    "cache_write_tokens": None,
                    "cacheWriteTokens": 2,
                },
            },
            2,
        ),
    ],
)
async def test_missing_and_null_upstream_usage_counters(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    streaming: bool,
    usage: dict[str, object],
    cache_write: int,
) -> None:
    upstream: dict[str, object] = {**_RESPONSE_OBJECT, "usage": usage}
    route: respx.Route = mock_core.post(_CORE_PATH)
    if streaming:
        route.respond(
            content=f"data: {json.dumps({'choices': [], 'usage': usage})}\n\ndata: [DONE]\n\n",
            content_type="text/event-stream",
        )
    else:
        route.respond(json=upstream)
    response: httpx.Response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "stream": streaming}
    )
    assert response.status_code == 200
    if streaming:
        counters = next(
            event["usage"]
            for name, event in parse_anthropic_sse(response.content)
            if name == "message_delta"
        )
    else:
        counters = response.json()["usage"]
    assert counters["input_tokens"] == 0
    assert counters["output_tokens"] == 0
    assert counters["cache_creation_input_tokens"] == cache_write


async def test_invalid_upstream_response_returns_generic_server_error(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_core.post(_CORE_PATH).respond(json={**_RESPONSE_OBJECT, "id": 42})
    with caplog.at_level(logging.ERROR, logger="bedrock"):
        response: httpx.Response = await client.post(
            _MESSAGES_URL, json=_MESSAGES_BODY
        )
    assert response.status_code == 500
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }
    assert any(record.exc_info is not None for record in caplog.records)


async def test_invalid_content_source_returns_client_error(
    client: httpx.AsyncClient,
) -> None:
    response: httpx.Response = await client.post(
        _MESSAGES_URL,
        json={
            **_MESSAGES_BODY,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": "invalid"}],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "messages.0.content.str: Input should be a valid string; messages.0.content.list[ContentBlock].0.source: Input should be a valid dictionary or instance of ContentSource",
        },
    }
