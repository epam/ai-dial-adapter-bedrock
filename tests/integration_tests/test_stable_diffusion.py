import base64
from typing import Dict
from unittest.mock import patch

import pytest

from aidial_adapter_bedrock.aws_client_config import (
    AWSClientConfigFactory,
    UpstreamConfig,
)
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.integration_tests.constants import BLUE_PNG_PICTURE
from tests.utils.mock_storage import MockFileStorage
from tests.utils.openai import (
    user,
    user_with_attachment_data,
    user_with_image_content_part,
)


def get_upstream_headers(region: str) -> Dict[str, str]:
    return {
        AWSClientConfigFactory.UPSTREAM_CONFIG_HEADER_NAME: UpstreamConfig(
            region=region
        ).json()
    }


def is_base64(s: str) -> bool:
    try:
        base64.b64decode(s)
        return True
    except Exception:
        return False


@pytest.fixture
def mock_v3_storage():
    storage = MockFileStorage.create()
    with patch(
        "aidial_adapter_bedrock.llm.model.stabililty.v3.create_file_storage",
        return_value=storage,
    ):
        yield storage
        storage.cleanup()


@pytest.mark.parametrize(
    "deployment, region, messages",
    [
        (
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1,
            "us-west-2",
            [user("generate image of a dog")],
        ),
        (
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1,
            "us-west-2",
            [
                user_with_image_content_part(
                    "generate image of a dog", BLUE_PNG_PICTURE
                )
            ],
        ),
        (
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1,
            "us-west-2",
            [
                user_with_attachment_data(
                    "generate image of a dog", BLUE_PNG_PICTURE
                )
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_stable_diffusion_v3(
    get_openai_client, mock_v3_storage, deployment, region, messages
):
    deployment = ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1
    REGION = "us-west-2"  # Stability models are available in us-west-2

    client = get_openai_client(deployment.value, get_upstream_headers(REGION))

    messages = [user("generate image of a dog")]

    response = await client.chat.completions.create(
        model=deployment.value,
        messages=messages,
        max_tokens=None,
    )

    assert len(response.choices) > 0
    choice = response.choices[0]

    # The response should contain a message with base64-encoded image data
    assert choice.message.content is not None
    assert choice.message.custom_content is not None
    assert len(choice.message.custom_content["attachments"]) == 1
    attachment = choice.message.custom_content["attachments"][0]
    assert attachment["type"] == "image/png"
    assert attachment["url"] is not None

    assert is_base64(await mock_v3_storage.download_file(attachment["url"]))
