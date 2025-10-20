import base64
from typing import Callable, List
from unittest.mock import patch

import openai
import pytest
from openai import APIStatusError, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.utils.resource import Resource
from tests.integration_tests.constants import BLUE_PNG_PICTURE, DOG_PICTURE
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
    (ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1, _WEST),
    (ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_5_LARGE_V1, _WEST),
]
IMAGE_GENERATION_MODELS = (
    TEXT_TO_IMAGE_ONLY_MODELS + IMAGE_TO_IMAGE_SUPPORTED_MODELS
)

VISION_MODEL = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET.US


def is_base64(s: bytes) -> bool:
    try:
        base64.b64decode(s)
        return True
    except Exception:
        return False


def _validate_attachment_url(response: ChatCompletion) -> str:
    assert len(response.choices) > 0
    choice = response.choices[0]

    assert choice.message.content is not None
    cc = choice.message.custom_content  # type: ignore

    assert cc is not None
    assert len(cc["attachments"]) == 1
    attachment = cc["attachments"][0]
    assert attachment["type"] == "image/png"
    assert attachment["url"] is not None
    return attachment["url"]


@pytest.fixture
def mock_storage():
    storage = MockFileStorage.create()
    with (
        patch(
            "aidial_adapter_bedrock.llm.model.stability.v1.create_file_storage",
            return_value=storage,
        ),
        patch(
            "aidial_adapter_bedrock.llm.model.stability.v2.create_file_storage",
            return_value=storage,
        ),
    ):
        yield storage
        storage.cleanup()


@pytest.fixture
def vision_model(get_openai_client):
    return get_openai_client(VISION_MODEL.value, region=_WEST)


@pytest.mark.parametrize("deployment, region", IMAGE_GENERATION_MODELS)
async def test_text_to_image(
    vision_model: AsyncAzureOpenAI,
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    mock_storage: FileStorage,
    deployment: ChatCompletionDeployment,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    response = await client.chat.completions.create(
        model=deployment.value,
        messages=[user("generate image of a dog")],
        max_tokens=None,
    )
    attachment_url = _validate_attachment_url(response)
    result_image_content = await mock_storage.download_file(attachment_url)

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
    assert "YES" in (vision_response.choices[0].message.content or "")


@pytest.mark.parametrize("deployment, region", TEXT_TO_IMAGE_ONLY_MODELS)
async def test_image_to_image_unsupported(
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: ChatCompletionDeployment,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)

    with pytest.raises(APIStatusError) as exc_info:
        await client.chat.completions.create(
            model=deployment.value,
            messages=[user_with_image_content_part("Brown dog", DOG_PICTURE)],
        )
    assert exc_info.value.status_code == 422
    assert "Image-to-Image is not supported" in exc_info.value.message


@pytest.mark.parametrize("deployment, region", IMAGE_TO_IMAGE_SUPPORTED_MODELS)
async def test_image_to_image_with_too_small_picture(
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: ChatCompletionDeployment,
    region: str,
):
    client = get_openai_client(deployment.value, region=region)
    with pytest.raises(APIStatusError) as exc_info:
        await client.chat.completions.create(
            model=deployment.value,
            messages=[user_with_image_content_part("test", BLUE_PNG_PICTURE)],
        )

    assert exc_info.value.status_code == 422
    assert (
        "Image width is 3, but should be between 640 and 1536"
        in exc_info.value.message
    )


@pytest.mark.parametrize(
    "message",
    [
        user_with_image_content_part(
            "Dog with red flowers in basket nearby", DOG_PICTURE
        ),
        user_with_attachment_data(
            "Dog with red flowers in basket nearby", DOG_PICTURE
        ),
    ],
)
@pytest.mark.parametrize("deployment, region", IMAGE_TO_IMAGE_SUPPORTED_MODELS)
async def test_image_to_image(
    vision_model: AsyncAzureOpenAI,
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    mock_storage: FileStorage,
    deployment: ChatCompletionDeployment,
    region: str,
    message: ChatCompletionMessageParam,
):

    client: AsyncAzureOpenAI = get_openai_client(
        deployment.value, region=region
    )
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
                    data=await mock_storage.download_file(attachment_url),
                ),
            )
        ],
    )
    assert "YES" in (vision_response.choices[0].message.content or "")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("deployment, region", IMAGE_GENERATION_MODELS)
async def test_content_filtering(
    get_openai_client: Callable[..., AsyncAzureOpenAI],
    deployment: ChatCompletionDeployment,
    region: str,
    stream: bool,
):
    client: AsyncAzureOpenAI = get_openai_client(
        deployment.value, region=region
    )
    messages: List[ChatCompletionMessageParam] = [
        user("generate an explicit image depicting copulating homo sapiens")
    ]

    with pytest.raises(Exception) as exc_info:
        await client.chat.completions.create(
            model=deployment.value,
            messages=messages,
            max_tokens=None,
            stream=stream,
        )

    assert isinstance(exc_info.value, openai.BadRequestError)

    resp = exc_info.value.response.json()
    assert resp["error"]["code"] == "content_filter"
    assert resp["error"]["message"] == "Filter reason: prompt"
