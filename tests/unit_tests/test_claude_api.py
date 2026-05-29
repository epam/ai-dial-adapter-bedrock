from pathlib import Path

import httpx
import pytest
import respx
from httpx import ASGITransport

from aidial_adapter_bedrock.claude_api import app


def _read_fixture(name: str) -> bytes:
    fixtures = Path(__file__).parent / "fixtures" / "claude_api"
    return (fixtures / name).read_bytes()


_MESSAGES_REQUEST = {
    "model": "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Say hello."}],
}


@pytest.fixture
def mock():
    with respx.mock(base_url=_ANTHROPIC_API) as mock:
        yield mock


_ANTHROPIC_API = "https://api.anthropic.com"


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=ASGITransport(app),  # type: ignore
        base_url="http://test-app.com",
        headers={"x-upstream-key": "test-claude-api-key"},
    ) as client:
        yield client


@respx.mock
async def test_messages_non_streaming(
    client: httpx.AsyncClient, mock: respx.MockRouter
):
    content = _read_fixture("messages_non_streaming_response.json")

    mock.post(url="/v1/messages").respond(
        content=content,
        content_type="application/json",
    )

    response = await client.post("/v1/messages", json=_MESSAGES_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "Hello! How can I assist you today?"


@respx.mock
async def test_messages_streaming(
    client: httpx.AsyncClient, mock: respx.MockRouter
):
    content = _read_fixture("messages_streaming_response.txt")
    mock.post("/v1/messages").respond(
        content=content,
        content_type="text/event-stream",
    )

    payload = {**_MESSAGES_REQUEST, "stream": True}
    response = await client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert b"message_start" in response.content
    assert b"Hello!" in response.content


@respx.mock
async def test_message_batches(
    client: httpx.AsyncClient, mock: respx.MockRouter
):
    content = _read_fixture("batches_response.json")

    mock.post(url="/v1/messages/batches").respond(
        content=content,
        content_type="application/json",
    )

    payload = {
        "requests": [
            {
                "custom_id": "req-1",
                "params": {**_MESSAGES_REQUEST},
            }
        ]
    }
    response = await client.post("/v1/messages/batches", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message_batch"
    assert body["processing_status"] == "in_progress"


@respx.mock
async def test_count_tokens(client: httpx.AsyncClient, mock: respx.MockRouter):
    content = _read_fixture("count_tokens_response.json")

    mock.post(url="/v1/messages/count_tokens").respond(
        content=content,
        content_type="application/json",
    )

    response = await client.post(
        "/v1/messages/count_tokens", json=_MESSAGES_REQUEST
    )

    assert response.status_code == 200
    body = response.json()
    assert body["input_tokens"] == 14


@respx.mock
async def test_models(client: httpx.AsyncClient, mock: respx.MockRouter):
    content = _read_fixture("models_response.json")

    mock.get(url="/v1/models").respond(
        content=content,
        content_type="application/json",
    )

    response = await client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["data"], list)
    assert len(body["data"]) == 2
    assert body["data"][0]["id"] == "claude-3-5-sonnet-20241022"
