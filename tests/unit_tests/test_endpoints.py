from typing import List, Tuple

import pytest
import requests

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.conftest import TEST_SERVER_URL

test_cases: List[Tuple[ChatCompletionDeployment, bool, bool]] = [
    (ChatCompletionDeployment.AMAZON_TITAN_TG1_LARGE, True, True),
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
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_US, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU_EU, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS, True, True),
    (ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS_US, True, True),
    (ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL, False, True),
    (ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL_V1, False, True),
    (ChatCompletionDeployment.META_LLAMA2_13B_CHAT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA2_70B_CHAT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_8B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_70B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_1_405B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.META_LLAMA3_1_8B_INSTRUCT_V1, True, True),
    (ChatCompletionDeployment.COHERE_COMMAND_TEXT_V14, True, True),
    (ChatCompletionDeployment.COHERE_COMMAND_LIGHT_TEXT_V14, True, True),
]


def feature_test_helper(
    url: str, is_supported: bool, headers: dict, payload: dict
) -> None:
    response = requests.post(url, json=payload, headers=headers)
    assert (
        response.status_code != 404
    ) == is_supported, (
        f"is_supported={is_supported}, code={response.status_code}, url={url}"
    )


@pytest.mark.parametrize(
    "deployment, tokenize_supported, truncate_supported", test_cases
)
def test_model_features(
    server,
    deployment: ChatCompletionDeployment,
    tokenize_supported: bool,
    truncate_supported: bool,
):
    payload = {"inputs": []}
    headers = {"Content-Type": "application/json", "Api-Key": "dummy"}

    BASE_URL = f"{TEST_SERVER_URL}/openai/deployments/{deployment.value}"

    tokenize_url = f"{BASE_URL}/tokenize"
    feature_test_helper(tokenize_url, tokenize_supported, headers, payload)

    truncate_url = f"{BASE_URL}/truncate_prompt"
    feature_test_helper(truncate_url, truncate_supported, headers, payload)
