import json

import httpx
import pytest

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from tests.utils.openai import user

_REGION = "us-east-1"


# https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-anthropic.html
_UPSTREAM_MANTLE_TOKENIZE_DEPLOYMENTS = [
    D.ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE.value,
    D.ANTHROPIC_CLAUDE_V4_7_OPUS.value,
    D.ANTHROPIC_CLAUDE_V4_8_OPUS.value,
    D.ANTHROPIC_CLAUDE_V5_SONNET.value,
    D.ANTHROPIC_CLAUDE_V5_OPUS.value,
    D.ANTHROPIC_CLAUDE_V5_FABLE.value,
]


@pytest.mark.parametrize(
    "deployment",
    _UPSTREAM_MANTLE_TOKENIZE_DEPLOYMENTS,
    ids=lambda deployment: deployment,
)
async def test_claude_mantle_upstream_tokenization(
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


_UPSTREAM_CONVERSE_TOKENIZE_DEPLOYMENTS = [
    D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
    D.ANTHROPIC_CLAUDE_V4_SONNET,
    D.ANTHROPIC_CLAUDE_V4_5_SONNET,
    D.ANTHROPIC_CLAUDE_V4_6_OPUS,
    D.ANTHROPIC_CLAUDE_V4_6_SONNET,
    D.ANTHROPIC_CLAUDE_V4_1_OPUS,
    D.ANTHROPIC_CLAUDE_V5_FABLE,
]


@pytest.mark.parametrize(
    "deployment",
    _UPSTREAM_CONVERSE_TOKENIZE_DEPLOYMENTS,
    ids=lambda deployment: deployment.value,
)
async def test_claude_converse_upstream_tokenization(
    test_http_client: httpx.AsyncClient,
    deployment: D,
):
    if deployment == D.ANTHROPIC_CLAUDE_V5_FABLE:
        # Bedrock rejects Fable 5 despite it is marked as supported in AWS docs:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5.html#model-card-anthropic-claude-fable-5-capabilities
        pytest.skip()

    response = await test_http_client.post(
        url=f"/openai/deployments/{deployment.value}/tokenize",
        json={
            "inputs": [
                {
                    "type": "request",
                    "value": {
                        "messages": [user("Count these tokens.")],
                        "custom_fields": {
                            "configuration": {"performanceConfig": {}}
                        },
                    },
                }
            ]
        },
        headers={
            "api-key": "dummy",
            "x-upstream-extra-data": json.dumps({"region": _REGION}),
        },
    )

    response.raise_for_status()
    output = response.json()["outputs"][0]
    assert output["status"] == "success"
    token_count = output["token_count"]
    assert isinstance(token_count, int)
    assert token_count > 0
