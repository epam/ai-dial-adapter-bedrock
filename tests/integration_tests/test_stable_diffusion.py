import base64
import os
from typing import Dict
from unittest.mock import patch

import pytest
from openai import APIStatusError

from aidial_adapter_bedrock.aws_client_config import (
    AWSClientConfigFactory,
    UpstreamConfig,
)
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.utils.resource import Resource
from tests.utils.mock_storage import MockFileStorage
from tests.utils.openai import (
    user,
    user_with_attachment_data,
    user_with_image_content_part,
)

_WEST = "us-west-2"
TEXT_TO_IMAGE_ONLY_MODELS = [
    (ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1, _WEST),
    (ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1, _WEST),
]

IMAGE_TO_IMAGE_SUPPORTED_MODELS = [
    (
        ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1,
        _WEST,
    ),
]
VISION_MODEL = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_US

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOG_IMAGE_PATH = os.path.join(
    CURRENT_DIR, "images", "dog-sample-image.png"
)

with open(SAMPLE_DOG_IMAGE_PATH, "rb") as f:
    SAMPLE_DOG_RESOURCE = Resource(
        type="image/png",
        data=f.read(),
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


def _validate_attachment_url(response) -> str:
    assert len(response.choices) > 0
    choice = response.choices[0]

    assert choice.message.content is not None
    assert choice.message.custom_content is not None
    assert len(choice.message.custom_content["attachments"]) == 1
    attachment = choice.message.custom_content["attachments"][0]
    assert attachment["type"] == "image/png"
    assert attachment["url"] is not None
    return attachment["url"]


@pytest.fixture
def mock_v3_storage():
    storage = MockFileStorage.create()
    with patch(
        "aidial_adapter_bedrock.llm.model.stability.v2.create_file_storage",
        return_value=storage,
    ):
        yield storage
        storage.cleanup()


@pytest.fixture
def vision_model(get_openai_client):
    return get_openai_client(VISION_MODEL.value, get_upstream_headers(_WEST))


@pytest.mark.parametrize(
    "deployment, region",
    [
        *TEXT_TO_IMAGE_ONLY_MODELS,
        *IMAGE_TO_IMAGE_SUPPORTED_MODELS,
    ],
)
@pytest.mark.asyncio
async def test_text_to_image(
    vision_model, get_openai_client, mock_v3_storage, deployment, region
):
    client = get_openai_client(deployment.value, get_upstream_headers(region))

    response = await client.chat.completions.create(
        model=deployment.value,
        messages=[user("generate image of a dog")],
        max_tokens=None,
    )
    attachment_url = _validate_attachment_url(response)
    result_image_content = await mock_v3_storage.download_file(attachment_url)

    assert is_base64(result_image_content)
    vision_response = await vision_model.chat.completions.create(
        model=VISION_MODEL.value,
        messages=[
            user_with_image_content_part(
                "Is there dog on the image? Answer only YES or NO.",
                Resource(
                    type="image/png",
                    data=result_image_content,
                ),
            )
        ],
    )
    assert "YES" in vision_response.choices[0].message.content


@pytest.mark.parametrize(
    "deployment, region",
    [*TEXT_TO_IMAGE_ONLY_MODELS],
)
@pytest.mark.asyncio
async def test_image_to_image_unsupported(
    get_openai_client,
    deployment,
    region,
):
    client = get_openai_client(deployment.value, get_upstream_headers(region))

    with pytest.raises(APIStatusError) as exc_info:
        await client.chat.completions.create(
            model=deployment.value,
            messages=[
                user_with_image_content_part("Brown dog", SAMPLE_DOG_RESOURCE)
            ],
        )
    assert exc_info.value.status_code == 422
    assert "Image-to-image is not supported" in exc_info.value.message


@pytest.mark.parametrize(
    "message",
    [
        user_with_image_content_part(
            "Dog with red flowers in basket nearby", SAMPLE_DOG_RESOURCE
        ),
        user_with_attachment_data(
            "Dog with red flowers in basket nearby", SAMPLE_DOG_RESOURCE
        ),
    ],
)
@pytest.mark.parametrize(
    "deployment, region",
    [*IMAGE_TO_IMAGE_SUPPORTED_MODELS],
)
@pytest.mark.asyncio
async def test_image_to_image(
    vision_model,
    get_openai_client,
    mock_v3_storage,
    deployment,
    region,
    message,
):

    client = get_openai_client(deployment.value, get_upstream_headers(region))
    response = await client.chat.completions.create(
        model=deployment.value,
        messages=[message],
        max_tokens=None,
    )
    attachment_url = _validate_attachment_url(response)
    vision_response = await vision_model.chat.completions.create(
        model=VISION_MODEL.value,
        messages=[
            user_with_image_content_part(
                "Is there dog and red flowers on the image? Answer only YES or NO.",
                Resource(
                    type="image/png",
                    data=await mock_v3_storage.download_file(attachment_url),
                ),
            )
        ],
    )
    assert "YES" in vision_response.choices[0].message.content
