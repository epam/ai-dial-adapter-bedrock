import anthropic
import httpx
import pytest
import respx

_CORE = "http://dial-core"


async def test_2_plus_3(
    monkeypatch: pytest.MonkeyPatch, test_http_client: httpx.AsyncClient
) -> None:
    monkeypatch.setenv("DIAL_URL", _CORE)

    with respx.mock(base_url=_CORE) as core:
        core.post("/openai/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "5"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

        async with anthropic.AsyncAnthropic(
            api_key="dummy-key",
            base_url="http://test-app.com/openaiChatCompletions/anthropic",
            http_client=test_http_client,
            max_retries=0,
        ) as client:
            response = await client.messages.create(
                model="gpt-5.5",
                max_tokens=100,
                messages=[{"role": "user", "content": "2+3=?"}],
            )

    assert response.content[0].type == "text"
    assert response.content[0].text == "5"
