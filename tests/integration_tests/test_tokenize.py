import json

import httpx
import pytest

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from tests.utils.openai import user

_REGION = "us-east-1"
_TOKENIZE_INPUTS = [
    {
        "type": "request",
        "value": {"messages": [user("Hello world!")]},
    },
    {"type": "string", "value": "Hello world!"},
]


_DEFAULT_CONVERSE_TOKENIZER_DEPLOYMENTS = [
    D.AMAZON_NOVA_PRO,
    D.AMAZON_NOVA_LITE,
    D.AMAZON_NOVA_MICRO,
    D.AI21_JAMBA_1_5_LARGE_V1,
    D.AI21_JAMBA_1_5_MINI_V1,
    D.ANTHROPIC_CLAUDE_V3_SONNET,
    D.ANTHROPIC_CLAUDE_V3_5_SONNET,
    D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
    D.ANTHROPIC_CLAUDE_V3_HAIKU,
    D.ANTHROPIC_CLAUDE_V3_5_HAIKU,
    D.ANTHROPIC_CLAUDE_V3_OPUS,
    D.ANTHROPIC_CLAUDE_V3_7_SONNET,
    D.ANTHROPIC_CLAUDE_V4_OPUS,
    D.ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE,
    D.ANTHROPIC_CLAUDE_V4_7_OPUS,
    D.ANTHROPIC_CLAUDE_V4_8_OPUS,
    D.ANTHROPIC_CLAUDE_V5_SONNET,
    D.ANTHROPIC_CLAUDE_V5_OPUS,
    D.META_LLAMA3_8B_INSTRUCT_V1,
    D.META_LLAMA3_70B_INSTRUCT_V1,
    D.META_LLAMA3_1_8B_INSTRUCT_V1,
    D.META_LLAMA3_1_70B_INSTRUCT_V1,
    D.META_LLAMA3_1_405B_INSTRUCT_V1,
    D.META_LLAMA3_2_1B_INSTRUCT_V1,
    D.META_LLAMA3_2_3B_INSTRUCT_V1,
    D.META_LLAMA3_2_11B_INSTRUCT_V1,
    D.META_LLAMA3_2_90B_INSTRUCT_V1,
    D.META_LLAMA3_3_70B_INSTRUCT_V1,
    D.META_LLAMA4_MAVERICK_17B_INSTRUCT_V1,
    D.META_LLAMA4_SCOUT_17B_INSTRUCT_V1,
    D.COHERE_COMMAND_R_V1,
    D.COHERE_COMMAND_R_PLUS_V1,
    D.DEEPSEEK_R1_V2,
    D.MINIMAX_M25,
]


def _assert_successful_tokenize_outputs(response: httpx.Response) -> None:
    response.raise_for_status()

    outputs = response.json()["outputs"]
    assert len(outputs) == len(_TOKENIZE_INPUTS)

    for output in outputs:
        assert output["status"] == "success"
        token_count = output["token_count"]
        assert isinstance(token_count, int)
        assert token_count > 0


# https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-anthropic.html
_MANTLE_DEPLOYMENTS_SUPPORT_TOKENIZE = [
    D.ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE.value,
    D.ANTHROPIC_CLAUDE_V4_7_OPUS.value,
    D.ANTHROPIC_CLAUDE_V4_8_OPUS.value,
    D.ANTHROPIC_CLAUDE_V5_SONNET.value,
    D.ANTHROPIC_CLAUDE_V5_OPUS.value,
    D.ANTHROPIC_CLAUDE_V5_FABLE.value,
]


@pytest.mark.parametrize(
    "deployment",
    _MANTLE_DEPLOYMENTS_SUPPORT_TOKENIZE,
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


@pytest.mark.parametrize(
    "deployment",
    _DEFAULT_CONVERSE_TOKENIZER_DEPLOYMENTS,
    ids=lambda deployment: deployment.value,
)
async def test_default_converse_tokenization_with_mixed_inputs(
    test_http_client: httpx.AsyncClient,
    deployment: D,
):
    response = await test_http_client.post(
        url=f"/openai/deployments/{deployment.value}/tokenize",
        json={"inputs": _TOKENIZE_INPUTS},
        headers={
            "api-key": "dummy",
            "x-upstream-extra-data": json.dumps({"region": _REGION}),
        },
    )

    _assert_successful_tokenize_outputs(response)


@pytest.mark.parametrize(
    "deployment",
    [
        D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V4_SONNET,
        D.ANTHROPIC_CLAUDE_V4_5_SONNET,
        D.ANTHROPIC_CLAUDE_V4_6_OPUS,
        D.ANTHROPIC_CLAUDE_V4_6_SONNET,
        D.ANTHROPIC_CLAUDE_V4_1_OPUS,
        D.ANTHROPIC_CLAUDE_V5_FABLE,
    ],
    ids=lambda deployment: deployment.value,
)
async def test_upstream_converse_tokenization_with_mixed_inputs(
    test_http_client: httpx.AsyncClient,
    deployment: D,
):
    if deployment == D.ANTHROPIC_CLAUDE_V5_FABLE:
        # Bedrock rejects Fable 5 despite it is marked as supported in AWS docs:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5.html#model-card-anthropic-claude-fable-5-capabilities
        pytest.skip()

    request_input, string_input = _TOKENIZE_INPUTS
    response = await test_http_client.post(
        url=f"/openai/deployments/{deployment.value}/tokenize",
        json={
            "inputs": [
                {
                    **request_input,
                    "value": {
                        **request_input["value"],
                        "custom_fields": {
                            "configuration": {"performanceConfig": {}}
                        },
                    },
                },
                string_input,
            ]
        },
        headers={
            "api-key": "dummy",
            "x-upstream-extra-data": json.dumps({"region": _REGION}),
        },
    )

    _assert_successful_tokenize_outputs(response)
