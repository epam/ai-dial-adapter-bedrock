import openai
import pytest
import respx

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.utils.openai import chat_completion, user


@respx.mock
@pytest.mark.parametrize("streaming", [False, True])
async def test_anthropic_error(get_openai_client, streaming: bool):
    deployment = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET_US
    region = "us-east-1"

    client: openai.AsyncAzureOpenAI = get_openai_client(deployment.value)
    client.max_retries = 0

    endpoint = "invoke-with-response-stream" if streaming else "invoke"

    respx.post(
        f"https://bedrock-runtime.{region}.amazonaws.com/model/{deployment.value}/{endpoint}",
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
