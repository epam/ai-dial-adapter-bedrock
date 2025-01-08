from typing import List, Tuple

import httpx
import pytest

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.utils.validation import check_enum_completeness

test_cases: List[Tuple[ChatCompletionDeployment, bool, bool]] = [
    (ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE, True, True),
    (ChatCompletionDeployment.AMAZON_NOVA_PRO, True, True),
    (ChatCompletionDeployment.AMAZON_NOVA_LITE, True, True),
    (ChatCompletionDeployment.AMAZON_NOVA_MICRO, True, True),
    (ChatCompletionDeployment.AI21_J2_GRANDE_INSTRUCT, True, True),
    (ChatCompletionDeployment.AI21_J2_JUMBO_INSTRUCT, True, True),
    (ChatCompletionDeployment.AI21_J2_MID_V1, True, True),
    (ChatCompletionDeployment.AI21_J2_ULTRA_V1, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_INSTANT_V1, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V2_1, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_US, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET_EU, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_EU, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2_US, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU_US, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US, True, True),
    (ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL, False, True),
    (ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL_V1, False, True),
    (
        ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1,
        False,
        True,
    ),
    (ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1, False, True),
    (ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1, False, True),
    (ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_2_1B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_2_3B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_2_11B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_2_90B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14, True, True),
    (ChatCompletionDeployment.COHERE_COMMAND_LIGHT_TEXT_V14, True, True),
]


check_enum_completeness([model for model, _, _ in test_cases])


async def assert_feature(
    http_client: httpx.AsyncClient,
    endpoint: str,
    is_supported: bool,
    headers: dict,
    payload: dict,
) -> None:
    response = await http_client.post(endpoint, json=payload, headers=headers)
    assert (
        response.status_code != 404
    ) == is_supported, f"is_supported={is_supported}, code={response.status_code}, url={endpoint}"


@pytest.mark.parametrize(
    "deployment, tokenize_supported, truncate_supported", test_cases
)
async def test_model_features(
    test_http_client: httpx.AsyncClient,
    deployment: ChatCompletionDeployment,
    tokenize_supported: bool,
    truncate_supported: bool,
):
    payload = {"inputs": []}
    headers = {"Content-Type": "application/json", "Api-Key": "dummy"}

    base = f"openai/deployments/{deployment.value}"

    tokenize_endpoint = f"{base}/tokenize"
    await assert_feature(
        test_http_client,
        tokenize_endpoint,
        tokenize_supported,
        headers,
        payload,
    )

    truncate_endpoint = f"{base}/truncate_prompt"
    await assert_feature(
        test_http_client,
        truncate_endpoint,
        truncate_supported,
        headers,
        payload,
    )
