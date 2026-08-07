import json

import httpx
import pytest

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from tests.utils.openai import user

_REGION = "us-east-1"
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-anthropic.html
_DEPLOYMENTS_SUPPORT_TOKENIZE = [
    D.ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE.value,
    D.ANTHROPIC_CLAUDE_V4_7_OPUS.value,
    D.ANTHROPIC_CLAUDE_V4_8_OPUS.value,
    D.ANTHROPIC_CLAUDE_V5_SONNET.value,
    D.ANTHROPIC_CLAUDE_V5_OPUS.value,
    D.ANTHROPIC_CLAUDE_V5_FABLE.value,
]


@pytest.mark.parametrize(
    "deployment",
    _DEPLOYMENTS_SUPPORT_TOKENIZE,
    ids=lambda deployment: deployment,
)
async def test_claude_upstream_tokenization(
    test_http_client: httpx.AsyncClient,
    deployment: str,
):
    response = await test_http_client.post(
        url=f"/openai/deployments/{deployment}/tokenize",
        json={
            "inputs": [
                {
                    "type": "request",
                    "value": {"messages": [user("Count these tokens.")]},
                }
            ]
        },
        headers={
            "api-key": "dummy",
            "x-upstream-extra-data": json.dumps(
                {
                    "region": _REGION,
                    "claude_client": "mantle",
                }
            ),
        },
    )

    response.raise_for_status()
    output = response.json()["outputs"][0]
    assert output["status"] == "success"
    token_count = output["token_count"]
    assert isinstance(token_count, int)
    assert token_count > 0
