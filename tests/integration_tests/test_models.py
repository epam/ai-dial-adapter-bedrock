from typing import List

import pytest
from openai import AsyncAzureOpenAI

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment


async def models_request_openai(client: AsyncAzureOpenAI) -> List[str]:
    data = (await client.models.list()).data
    return [model.id for model in data]


def assert_models_subset(actual_models: List[str]):
    expected_models = [option.value for option in ChatCompletionDeployment]

    assert set(expected_models).issubset(
        set(actual_models)
    ), f"Expected models: {expected_models}, Actual models: {actual_models}"


@pytest.mark.asyncio
async def test_model_list_openai(get_openai_client):
    assert_models_subset(await models_request_openai(get_openai_client()))
