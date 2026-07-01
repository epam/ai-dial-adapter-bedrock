import gzip
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import anthropic
import httpx
import pytest
import respx
from anthropic.types import MessageParam
from httpx import ASGITransport

from aidial_adapter_bedrock.claude_api import app


def _read_fixture(name: str) -> bytes:
    fixtures = Path(__file__).parent / "fixtures" / "claude_api"
    return (fixtures / name).read_bytes()


class _BaseMessagesRequest(TypedDict):
    model: str
    messages: Iterable[MessageParam]


class _MessagesRequest(_BaseMessagesRequest):
    max_tokens: int


_BASE_MESSAGES_REQUEST: _BaseMessagesRequest = {
    "model": "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "messages": [{"role": "user", "content": "Say hello."}],
}

_MESSAGES_REQUEST: _MessagesRequest = {
    **_BASE_MESSAGES_REQUEST,
    "max_tokens": 1024,
}


@pytest.fixture
def mock():
    with respx.mock(base_url="https://api.anthropic.com") as mock:
        yield mock


@pytest.fixture
async def http_client():
    async with httpx.AsyncClient(
        transport=ASGITransport(app),  # type: ignore
        base_url="http://test-app.com",
        headers={"x-upstream-key": "test-claude-api-key"},
    ) as client:
        yield client


@pytest.fixture
async def anthropic_client(http_client: httpx.AsyncClient):
    async with anthropic.AsyncAnthropic(
        api_key="test-claude-api-key",
        http_client=http_client,
        max_retries=0,
    ) as client:
        yield client


class TestMessagesNonStreaming:
    @pytest.fixture(autouse=True)
    def _setup(self, mock: respx.MockRouter):
        content = _read_fixture("messages_non_streaming_response.json")
        mock.post(url="/v1/messages").respond(
            content=content, content_type="application/json"
        )

    @respx.mock
    async def test_http(self, http_client: httpx.AsyncClient):
        response = await http_client.post(
            "/v1/messages", json=_MESSAGES_REQUEST
        )

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert (
            body["content"][0]["text"] == "Hello! How can I assist you today?"
        )

    @respx.mock
    async def test_anthropic(self, anthropic_client: anthropic.AsyncAnthropic):
        response = await anthropic_client.messages.create(**_MESSAGES_REQUEST)

        assert response.type == "message"
        assert response.role == "assistant"
        text_block = response.content[0]
        assert isinstance(text_block, anthropic.types.TextBlock)
        assert text_block.text == "Hello! How can I assist you today?"

    @respx.mock
    async def test_http_gzip_encoding_stripped(
        self, mock: respx.MockRouter, http_client: httpx.AsyncClient
    ):
        content = _read_fixture("messages_non_streaming_response.json")
        mock.post(url="/v1/messages").respond(
            content=gzip.compress(content),
            content_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )

        response = await http_client.post(
            "/v1/messages",
            json=_MESSAGES_REQUEST,
            headers={"Accept-Encoding": "gzip"},
        )

        assert response.status_code == 200
        assert "content-encoding" not in response.headers
        body = response.json()
        assert body["type"] == "message"


class TestMessagesStreaming:
    @pytest.fixture(autouse=True)
    def _setup(self, mock: respx.MockRouter):
        content = _read_fixture("messages_streaming_response.txt")
        mock.post("/v1/messages").respond(
            content=content, content_type="text/event-stream"
        )

    @respx.mock
    async def test_http(self, http_client: httpx.AsyncClient):
        payload = {**_MESSAGES_REQUEST, "stream": True}
        response = await http_client.post("/v1/messages", json=payload)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert b"message_start" in response.content
        for txt in ["Hello!", " How can I", " assist you today?"]:
            assert txt.encode() in response.content

    @respx.mock
    async def test_anthropic(self, anthropic_client: anthropic.AsyncAnthropic):
        async with anthropic_client.messages.stream(
            **_MESSAGES_REQUEST
        ) as stream:
            text = await stream.get_final_text()

        assert text == "Hello! How can I assist you today?"


class TestMessageBatches:
    @pytest.fixture(autouse=True)
    def _setup(self, mock: respx.MockRouter):
        content = _read_fixture("batches_response.json")
        mock.post(url="/v1/messages/batches").respond(
            content=content, content_type="application/json"
        )

    @respx.mock
    async def test_http(self, http_client: httpx.AsyncClient):
        payload = {
            "requests": [
                {
                    "custom_id": "req-1",
                    "params": {**_MESSAGES_REQUEST},
                }
            ]
        }
        response = await http_client.post("/v1/messages/batches", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message_batch"
        assert body["processing_status"] == "in_progress"

    @respx.mock
    async def test_anthropic(self, anthropic_client: anthropic.AsyncAnthropic):
        batch = await anthropic_client.messages.batches.create(
            requests=[
                {
                    "custom_id": "req-1",
                    "params": _MESSAGES_REQUEST,  # type: ignore
                }
            ]
        )

        assert batch.type == "message_batch"
        assert batch.processing_status == "in_progress"


class TestCountTokens:
    @pytest.fixture(autouse=True)
    def _setup(self, mock: respx.MockRouter):
        content = _read_fixture("count_tokens_response.json")
        mock.post(url="/v1/messages/count_tokens").respond(
            content=content, content_type="application/json"
        )

    @respx.mock
    async def test_http(self, http_client: httpx.AsyncClient):
        response = await http_client.post(
            "/v1/messages/count_tokens", json=_BASE_MESSAGES_REQUEST
        )

        assert response.status_code == 200
        body = response.json()
        assert body["input_tokens"] == 14

    @respx.mock
    async def test_anthropic(self, anthropic_client: anthropic.AsyncAnthropic):
        response = await anthropic_client.messages.count_tokens(
            **_BASE_MESSAGES_REQUEST
        )

        assert response.input_tokens == 14


class TestDebugLogging:
    # The logs must have no effect unless the log level is DEBUG.
    @pytest.mark.parametrize(
        "level, expected", [(logging.DEBUG, True), (logging.INFO, False)]
    )
    @respx.mock
    async def test_request_and_response_logging(
        self,
        mock: respx.MockRouter,
        http_client: httpx.AsyncClient,
        caplog: pytest.LogCaptureFixture,
        level: int,
        expected: bool,
    ):
        content = _read_fixture("messages_non_streaming_response.json")
        mock.post(url="/v1/messages").respond(
            content=content, content_type="application/json"
        )

        with caplog.at_level(level, logger="bedrock"):
            response = await http_client.post(
                "/v1/messages", json=_MESSAGES_REQUEST
            )

        assert response.status_code == 200

        messages = [record.getMessage() for record in caplog.records]
        request_logs = [m for m in messages if m.startswith("request: ")]
        response_logs = [m for m in messages if m.startswith("response: ")]

        if expected:
            assert len(request_logs) == 1
            assert "Say hello." in request_logs[0]

            assert len(response_logs) == 1
            assert "Hello! How can I assist you today?" in response_logs[0]
        else:
            assert not request_logs
            assert not response_logs

    @respx.mock
    async def test_streaming_response_chunk_logging(
        self,
        mock: respx.MockRouter,
        http_client: httpx.AsyncClient,
        caplog: pytest.LogCaptureFixture,
    ):
        content = _read_fixture("messages_streaming_response.txt")
        mock.post("/v1/messages").respond(
            content=content, content_type="text/event-stream"
        )

        with caplog.at_level(logging.DEBUG, logger="bedrock"):
            payload = {**_MESSAGES_REQUEST, "stream": True}
            response = await http_client.post("/v1/messages", json=payload)

        assert response.status_code == 200

        messages = [record.getMessage() for record in caplog.records]
        chunk_logs = [m for m in messages if m.startswith("response chunk: ")]

        # The streamed chunks are logged as-is; the number of chunks isn't
        # deterministic, so we only assert the streamed content is logged.
        assert chunk_logs
        streamed = "\n".join(chunk_logs)
        assert "message_start" in streamed
        assert "Hello!" in streamed
        assert "message_stop" in streamed

    @pytest.mark.parametrize(
        "level, expected_encoding",
        [(logging.DEBUG, "identity"), (logging.INFO, None)],
    )
    @respx.mock
    async def test_debug_disables_upstream_compression(
        self,
        mock: respx.MockRouter,
        http_client: httpx.AsyncClient,
        caplog: pytest.LogCaptureFixture,
        level: int,
        expected_encoding: str | None,
    ):
        # With debug logging on, the upstream is asked not to compress the
        # response (Accept-Encoding: identity) so it can be logged as-is.
        # Otherwise the client's default (compressed) negotiation is left alone.
        captured: dict[str, str | None] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["accept-encoding"] = request.headers.get("accept-encoding")
            content = _read_fixture("messages_non_streaming_response.json")
            return httpx.Response(
                200,
                content=content,
                headers={"content-type": "application/json"},
            )

        mock.post(url="/v1/messages").mock(side_effect=_handler)

        with caplog.at_level(level, logger="bedrock"):
            response = await http_client.post(
                "/v1/messages", json=_MESSAGES_REQUEST
            )

        assert response.status_code == 200
        if expected_encoding is None:
            assert captured["accept-encoding"] != "identity"
        else:
            assert captured["accept-encoding"] == expected_encoding


class TestModels:
    @pytest.fixture(autouse=True)
    def _setup(self, mock: respx.MockRouter):
        content = _read_fixture("models_response.json")
        mock.get(url="/v1/models").respond(
            content=content, content_type="application/json"
        )

    @respx.mock
    async def test_http(self, http_client: httpx.AsyncClient):
        response = await http_client.get("/v1/models")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body["data"], list)
        assert len(body["data"]) == 2
        assert body["data"][0]["id"] == "claude-3-5-sonnet-20241022"

    @respx.mock
    async def test_anthropic(self, anthropic_client: anthropic.AsyncAnthropic):
        models = await anthropic_client.models.list()

        assert len(models.data) == 2
        assert models.data[0].id == "claude-3-5-sonnet-20241022"
