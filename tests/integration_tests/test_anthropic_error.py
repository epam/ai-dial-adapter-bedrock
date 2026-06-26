import json

import openai
import pytest
import respx

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.utils.openai import chat_completion, user

_DEPLOYMENT = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET.US
_REGION = "us-east-1"


@respx.mock
@pytest.mark.parametrize("streaming", [False, True])
async def test_anthropic_error_immediate(get_openai_client, streaming: bool):
    client: openai.AsyncAzureOpenAI = get_openai_client(
        _DEPLOYMENT.value, region=_REGION
    )
    client.max_retries = 0

    endpoint = "invoke-with-response-stream" if streaming else "invoke"

    respx.post(
        f"https://bedrock-runtime.{_REGION}.amazonaws.com/model/{_DEPLOYMENT.value}/{endpoint}",
    ).respond(status_code=429, json={"message": "Too Many Requests"})

    with pytest.raises(Exception) as exc_info:
        await chat_completion(client, messages=[user("test")], stream=streaming)

    exc = exc_info.value

    assert isinstance(exc, openai.RateLimitError)
    assert exc.status_code == 429
    assert exc.body == {"code": "429", "message": "Too Many Requests"}


@respx.mock
@pytest.mark.parametrize("streaming", [True])
async def test_anthropic_error_streaming(get_openai_client, streaming: bool):
    client: openai.AsyncAzureOpenAI = get_openai_client(
        _DEPLOYMENT.value, region=_REGION
    )
    client.max_retries = 0

    endpoint = "invoke-with-response-stream" if streaming else "invoke"

    respx.post(
        f"https://bedrock-runtime.{_REGION}.amazonaws.com/model/{_DEPLOYMENT.value}/{endpoint}",
    ).respond(
        status_code=200,
        content=b"\x00\x00\x00\xc2\x00\x00\x00a\xccB5\x1c\x0f:exception-type\x07\x00\x13"
        + b"validationException\r:content-type\x07\x00\x10application/json\r:message-type\x07\x00\t"
        + b'exception{"message":"messages.1.content.1.text.citations: Extra inputs are not permitted"}\x18+\x9f\xf9',
        headers={"Content-Type": "text/event-stream"},
    )

    with pytest.raises(Exception) as exc_info:
        await chat_completion(client, messages=[user("test")], stream=streaming)

    exc = exc_info.value

    assert isinstance(exc, openai.BadRequestError)
    assert exc.status_code == 400
    assert exc.body == {
        "code": "400",
        "type": "invalid_request_error",
        "message": "messages.1.content.1.text.citations: Extra inputs are not permitted",
    }


@respx.mock
@pytest.mark.parametrize("streaming", [False, True])
async def test_anthropic_error_rate_limit(
    get_openai_client,
    streaming: bool,
):
    client: openai.AsyncAzureOpenAI = get_openai_client(
        _DEPLOYMENT.value, region=_REGION
    )
    client.max_retries = 0

    endpoint = "invoke-with-response-stream" if streaming else "invoke"

    respx.post(
        f"https://bedrock-runtime.{_REGION}.amazonaws.com/model/{_DEPLOYMENT.value}/{endpoint}",
    ).respond(
        status_code=429,
        json={
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": (
                    "This request would exceed the rate limit for your organization "
                    "(test-org-id) of 30,000 input tokens per minute."
                ),
            },
            "request_id": "req_test_123",
        },
        headers={
            "Content-Type": "application/json",
            "Retry-After": "9",
        },
    )

    with pytest.raises(openai.RateLimitError) as exc_info:
        await chat_completion(
            client,
            messages=[user("test")],
            stream=streaming,
        )

    exc = exc_info.value

    assert exc.status_code == 429
    assert exc.response.headers is not None
    assert exc.response.headers.get("Retry-After") == "9"


async def test_openai_invalid_client_selector_returns_500(get_openai_client):
    client: openai.AsyncAzureOpenAI = get_openai_client(
        _DEPLOYMENT.value,
        extra_headers={
            "x-upstream-extra-data": json.dumps(
                {"region": _REGION, "client": "invalid"}
            )
        },
    )
    client.max_retries = 0

    with pytest.raises(openai.InternalServerError) as exc_info:
        await chat_completion(client, messages=[user("test")], stream=False)

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.body is not None
    assert isinstance(exc.body, dict)
    assert "validation error for UpstreamConfigData" in str(
        exc.body.get("message", "")
    )
    assert "client" in str(exc.body.get("message", ""))
    assert "legacy" in str(exc.body.get("message", ""))
    assert "mantle" in str(exc.body.get("message", ""))
    assert str(exc.body.get("type", "")) == "internal_server_error"
