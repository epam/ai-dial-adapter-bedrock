import openai
import pytest
import respx

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.utils.openai import chat_completion, user

_DEPLOYMENT = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET_US
_REGION = "us-east-1"


@respx.mock
@pytest.mark.parametrize("streaming", [False, True])
async def test_anthropic_error_immediate(get_openai_client, streaming: bool):
    client: openai.AsyncAzureOpenAI = get_openai_client(_DEPLOYMENT.value)
    client.max_retries = 0

    endpoint = "invoke-with-response-stream" if streaming else "invoke"

    respx.post(
        f"https://bedrock-runtime.{_REGION}.amazonaws.com/model/{_DEPLOYMENT.value}/{endpoint}",
    ).respond(status_code=429, json={"message": "Too Many Requests"})

    with pytest.raises(Exception) as exc_info:
        await chat_completion(
            client,
            [user("test")],
            streaming,
            None,
            None,
            None,
            None,
            None,
            0,
        )

    exc = exc_info.value

    assert isinstance(exc, openai.RateLimitError)
    assert exc.status_code == 429
    assert exc.body == {"code": "429", "message": "Too Many Requests"}


@respx.mock
@pytest.mark.parametrize("streaming", [True])
async def test_anthropic_error_streaming(get_openai_client, streaming: bool):
    client: openai.AsyncAzureOpenAI = get_openai_client(_DEPLOYMENT.value)
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
        await chat_completion(
            client,
            [user("test")],
            streaming,
            None,
            None,
            None,
            None,
            None,
            0,
        )

    exc = exc_info.value

    assert isinstance(exc, openai.BadRequestError)
    assert exc.status_code == 400
    assert exc.body == {
        "code": "400",
        "type": "invalid_request_error",
        "message": "messages.1.content.1.text.citations: Extra inputs are not permitted",
    }
