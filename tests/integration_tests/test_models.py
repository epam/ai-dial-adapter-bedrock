from typing import Any

import openai
import openai.error
import requests

from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from tests.conftest import DEFAULT_API_VERSION, TEST_SERVER_URL


def models_request_http() -> Any:
    response = requests.get(f"{TEST_SERVER_URL}/openai/models")
    assert response.status_code == 200
    return response.json()


def models_request_openai() -> Any:
    return openai.Model.list(
        api_type="azure",
        api_base=TEST_SERVER_URL,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )


def assert_models_subset(models: Any):
    actual_models = [model["id"] for model in models["data"]]
    expected_models = [option.value for option in BedrockDeployment]

    assert set(expected_models).issubset(
        set(actual_models)
    ), f"Expected models: {expected_models}, Actual models: {actual_models}"


def test_model_list_http(server):
    assert_models_subset(models_request_http())


def test_model_list_openai(server):
    assert_models_subset(models_request_openai())
