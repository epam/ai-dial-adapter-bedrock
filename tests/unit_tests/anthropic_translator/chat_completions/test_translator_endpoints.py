import json
import logging

import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from httpx import ASGITransport

from aidial_adapter_bedrock.anthropic_translator.capabilities import clear_cache
from tests.unit_tests.anthropic_translator.helpers import catalog

_MESSAGES_URL = "/to-chat-completions/anthropic/v1/messages"
_COUNT_TOKENS_URL = "/to-chat-completions/anthropic/v1/messages/count_tokens"
_CORE = "http://dial-core"
_MODELS_PATH = "/openai/models"
# Chat Completions is addressed per-deployment; the SDK appends
# `/chat/completions` and an `api-version` query param.
_CORE_PATH = "/openai/deployments/gpt-5.5/chat/completions"

_RESPONSE_OBJECT = {
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

_MESSAGES_BODY = {
    "model": "gpt-5.5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.fixture(autouse=True)
def _isolated_catalog_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setenv("DIAL_URL", _CORE)
    from aidial_adapter_bedrock.app import app

    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=ASGITransport(app),  # type: ignore
            base_url="http://test-app.com",
            headers={"api-key": "dummy-key"},
        ) as c,
    ):
        yield c


@pytest.fixture
def mock_core():
    """Core with a model catalog that lists `gpt-5.5` with no special
    capabilities; individual tests re-register the catalog route to change it.
    """
    with respx.mock(base_url=_CORE, assert_all_called=False) as mock:
        mock.get(_MODELS_PATH).respond(json=catalog())
        yield mock


async def test_non_streaming_happy_path(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    response = await client.post(_MESSAGES_URL, json=_MESSAGES_BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "hi there"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 7


async def test_request_shape_and_headers_not_leaked(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    route = mock_core.post(_CORE_PATH).respond(
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
    sent = route.calls.last.request
    sent_body = json.loads(sent.content)
    assert sent_body["model"] == "gpt-5.5"
    # The catalog doesn't advertise the newer spelling.
    assert sent_body["max_tokens"] == 100
    assert "max_completion_tokens" not in sent_body
    # `store` is never emitted (some adapters 400 on unknown top-level keys).
    assert "store" not in sent_body
    # Anthropic-specific headers must not leak to Core.
    assert "anthropic-version" not in sent.headers
    assert "anthropic-beta" not in sent.headers
    assert sent.headers["api-key"] == "dummy-key"
    assert "authorization" not in sent.headers


async def test_the_catalog_shapes_the_outbound_body(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    mock_core.get(_MODELS_PATH).respond(
        json=catalog(
            features={
                "max_completion_tokens_supported": True,
                "temperature": False,
                "reasoning_efforts": ["low", "medium", "high"],
            },
            limits={"max_completion_tokens": 64},
        )
    )
    route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    await client.post(
        _MESSAGES_URL,
        json={
            **_MESSAGES_BODY,
            "temperature": 0.5,
            "output_config": {"effort": "high"},
        },
    )
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["max_completion_tokens"] == 64  # clamped down
    assert "max_tokens" not in sent_body
    assert "temperature" not in sent_body
    assert sent_body["reasoning_effort"] == "high"


async def test_a_thinking_deployment_gets_the_nested_budget(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    """★ A deployment declaring `configuration.thinking` must never receive
    `reasoning_effort`."""
    mock_core.get(_MODELS_PATH).respond(
        json=catalog(
            features={"reasoning_efforts": ["low", "medium", "high"]},
            defaults={
                "custom_fields": {
                    "configuration": {"thinking": {"include_thoughts": True}}
                }
            },
        )
    )
    route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    await client.post(
        _MESSAGES_URL,
        json={
            **_MESSAGES_BODY,
            "output_config": {"effort": "medium"},
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        },
    )
    sent_body = json.loads(route.calls.last.request.content)
    assert "reasoning_effort" not in sent_body
    assert sent_body["custom_fields"]["configuration"]["thinking"] == {
        "include_thoughts": True,
        "thinking_budget": 4096,
    }


@pytest.mark.parametrize(
    "failure",
    [
        {"side_effect": httpx.ConnectError("refused")},
        {"side_effect": httpx.ReadTimeout("hangs")},
        {"return_value": httpx.Response(500)},
    ],
)
async def test_an_unreachable_catalog_still_answers_the_request(
    client: httpx.AsyncClient, mock_core: respx.MockRouter, failure
):
    """★ A catalog fetch that fails, 500s or hangs must not fail the user's
    message; it degrades to asserting nothing."""
    mock_core.get(_MODELS_PATH).mock(**failure)
    route = mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    response = await client.post(
        _MESSAGES_URL,
        json={**_MESSAGES_BODY, "output_config": {"effort": "high"}},
    )
    assert response.status_code == 200
    sent_body = json.loads(route.calls.last.request.content)
    assert "reasoning_effort" not in sent_body
    assert "custom_fields" not in sent_body


async def test_a_long_mcp_tool_name_round_trips(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    long_name = "mcp__" + "s" * 60 + "__do_the_thing"
    route = mock_core.post(_CORE_PATH).mock(
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
                                        # Echo back whatever alias was sent.
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
    response = await client.post(
        _MESSAGES_URL,
        json={**_MESSAGES_BODY, "tools": [{"name": long_name}]},
    )

    assert response.status_code == 200
    sent_alias = json.loads(route.calls.last.request.content)["tools"][0][
        "function"
    ]["name"]
    assert sent_alias != long_name
    assert len(sent_alias) <= 64
    # The client only ever sees the name it sent.
    assert response.json()["content"][0]["name"] == long_name


async def test_stop_is_stripped_and_emulated_for_a_gpt_5_deployment(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    route = mock_core.post(_CORE_PATH).respond(
        json={
            **_RESPONSE_OBJECT,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "keep STOP go"},
                    "finish_reason": "stop",
                }
            ],
        },
        content_type="application/json",
    )
    response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "stop_sequences": ["STOP"]}
    )

    assert "stop" not in json.loads(route.calls.last.request.content)
    body = response.json()
    assert body["content"][0]["text"] == "keep "
    assert body["stop_reason"] == "stop_sequence"
    assert body["stop_sequence"] == "STOP"


async def test_streaming_happy_path(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    upstream_sse = (
        b'data: {"id": "chatcmpl_1", "model": "gpt-5.5", "choices": '
        b'[{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, '
        b'"finish_reason": null}]}\n\n'
        b'data: {"id": "chatcmpl_1", "choices": [{"index": 0, "delta": {}, '
        b'"finish_reason": "stop"}]}\n\n'
        b'data: {"id": "chatcmpl_1", "choices": [], "usage": '
        b'{"prompt_tokens": 5, "completion_tokens": 2}}\n\n'
        b"data: [DONE]\n\n"
    )
    route = mock_core.post(_CORE_PATH).respond(
        content=upstream_sse, content_type="text/event-stream"
    )
    response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "stream": True}
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    sent_body = json.loads(route.calls.last.request.content)
    assert sent_body["stream"] is True
    # include_usage must be requested so the terminal usage is populated.
    assert sent_body["stream_options"] == {"include_usage": True}
    text = response.text
    assert "event: message_start" in text
    assert "event: ping" in text
    # `format_sse` uses `model_dump_json()` (compact separators, no space
    # after `:`) rather than stdlib `json.dumps`.
    assert '"text":"Hello"' in text
    assert "event: message_stop" in text


async def test_x_dial_deployment_id_header_overrides_body_model(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    # x-dial-deployment-id names the deployment Core actually routed to,
    # which can differ from the client-supplied `model` field.
    route = mock_core.post(
        "/openai/deployments/actual-deployment/chat/completions"
    ).respond(json=_RESPONSE_OBJECT, content_type="application/json")

    response = await client.post(
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
):
    response = await client.post(
        _MESSAGES_URL,
        json={
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "model" in body["error"]["message"]


async def test_connection_error_to_core_returns_502(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    mock_core.post(_CORE_PATH).mock(side_effect=httpx.ConnectError("refused"))
    response = await client.post(_MESSAGES_URL, json=_MESSAGES_BODY)
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "api_error"


async def test_debug_logging_emits_request_and_response_lines(
    client: httpx.AsyncClient, mock_core: respx.MockRouter, caplog
):
    mock_core.post(_CORE_PATH).respond(
        json=_RESPONSE_OBJECT, content_type="application/json"
    )
    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        response = await client.post(_MESSAGES_URL, json=_MESSAGES_BODY)

    assert response.status_code == 200
    messages = [r.getMessage() for r in caplog.records if r.name == "bedrock"]
    assert any(m.startswith("request: ") for m in messages)
    assert any(m.startswith("response: ") for m in messages)


async def test_debug_logging_emits_stream_chunk_lines(
    client: httpx.AsyncClient, mock_core: respx.MockRouter, caplog
):
    upstream_sse = (
        b'data: {"id": "chatcmpl_1", "model": "gpt-5.5", "choices": '
        b'[{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, '
        b'"finish_reason": "stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    mock_core.post(_CORE_PATH).respond(
        content=upstream_sse, content_type="text/event-stream"
    )
    with caplog.at_level(logging.DEBUG, logger="bedrock"):
        response = await client.post(
            _MESSAGES_URL, json={**_MESSAGES_BODY, "stream": True}
        )

    assert response.status_code == 200
    messages = [r.getMessage() for r in caplog.records if r.name == "bedrock"]
    assert any(m.startswith("response chunk: ") for m in messages)


async def test_missing_dial_url_returns_500(monkeypatch):
    monkeypatch.delenv("DIAL_URL", raising=False)
    from aidial_adapter_bedrock.app import app

    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=ASGITransport(app),  # type: ignore
            base_url="http://test-app.com",
        ) as c,
    ):
        response = await c.post(_MESSAGES_URL, json=_MESSAGES_BODY)

    assert response.status_code == 500
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert "DIAL_URL" in body["error"]["message"]


@pytest.mark.parametrize(
    "status, expected_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (503, "overloaded_error"),
        (529, "overloaded_error"),
    ],
)
async def test_upstream_error_status_mapping(
    client: httpx.AsyncClient,
    mock_core: respx.MockRouter,
    status: int,
    expected_type: str,
):
    route = mock_core.post(_CORE_PATH).respond(
        status_code=status,
        json={"error": {"message": "upstream says no"}},
    )
    response = await client.post(_MESSAGES_URL, json=_MESSAGES_BODY)

    assert response.status_code == status
    body = response.json()
    assert body["error"]["type"] == expected_type
    assert body["error"]["message"] == "upstream says no"
    # The SDK must not retry 429/5xx: the client sees the upstream status now.
    assert route.call_count == 1


async def test_pre_stream_error_returns_json_not_sse(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    # A 200 whose first frame is an error event is handled far worse by
    # clients than a proper HTTP status.
    mock_core.post(_CORE_PATH).respond(
        status_code=401, json={"error": {"message": "bad key"}}
    )
    response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "stream": True}
    )
    assert response.status_code == 401
    assert "application/json" in response.headers["content-type"]
    assert response.json()["error"]["type"] == "authentication_error"


async def test_malformed_json_returns_400(client: httpx.AsyncClient):
    response = await client.post(
        _MESSAGES_URL,
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_non_object_body_returns_400(client: httpx.AsyncClient):
    response = await client.post(_MESSAGES_URL, json=[1, 2, 3])
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_schema_violation_returns_400(client: httpx.AsyncClient):
    response = await client.post(
        _MESSAGES_URL, json={**_MESSAGES_BODY, "messages": "not-a-list"}
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "messages" in body["error"]["message"]


async def test_too_many_stop_sequences_returns_400(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    # `MessagesRequest.stop_sequences` has no length cap, but the outbound
    # `ChatCompletionRequest.stop` (`aidial_sdk`'s own `Stop` field) caps at 4
    # entries to match OpenAI's real limit. A well-formed Anthropic request
    # that exceeds it must surface as a 400 (client-fixable), not a 500 —
    # `translator_error_handler` reshapes the resulting `pydantic
    # .ValidationError` accordingly. `gpt-4o` is used because the `gpt-5.`
    # family omits `stop` entirely and so never hits the cap.
    mock_core.get(_MODELS_PATH).respond(json=catalog("gpt-4o"))
    response = await client.post(
        _MESSAGES_URL,
        json={
            **_MESSAGES_BODY,
            "model": "gpt-4o",
            "stop_sequences": ["a", "b", "c", "d", "e"],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_missing_max_tokens_returns_400(
    client: httpx.AsyncClient, mock_core: respx.MockRouter
):
    response = await client.post(
        _MESSAGES_URL,
        json={
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_count_tokens_path_returns_anthropic_404(
    client: httpx.AsyncClient,
):
    # Chat Completions has no token-counting endpoint, so this path is not
    # registered and falls through to the Anthropic-shaped catch-all 404.
    response = await client.post(_COUNT_TOKENS_URL, json=_MESSAGES_BODY)
    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"


async def test_unknown_path_returns_anthropic_404(client: httpx.AsyncClient):
    response = await client.post(
        "/to-chat-completions/anthropic/v1/messages/batches", json={}
    )
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"
