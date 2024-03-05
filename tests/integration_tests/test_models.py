from typing import List

import requests
from openai import AzureOpenAI

from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from tests.conftest import DEFAULT_API_VERSION, TEST_SERVER_URL


def models_request_http() -> List[str]:
    response = requests.get(f"{TEST_SERVER_URL}/openai/models")
    assert response.status_code == 200
    data = response.json()["data"]
    return [model["id"] for model in data]


def models_request_openai() -> List[str]:
    client = AzureOpenAI(
        azure_endpoint=TEST_SERVER_URL,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )
    data = client.models.list().data
    return [model.id for model in data]


def assert_models_subset(actual_models: List[str]):
    expected_models = [option.value for option in BedrockDeployment]

    assert set(expected_models).issubset(
        set(actual_models)
    ), f"Expected models: {expected_models}, Actual models: {actual_models}"


def test_model_list_http(server):
    assert_models_subset(models_request_http())


def test_model_list_openai(server):
    assert_models_subset(models_request_openai())
