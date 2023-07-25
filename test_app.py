from fastapi.testclient import TestClient

from app import app
from llm.bedrock_models import BedrockModels

client = TestClient(app)


def test_models():
    response = client.get("/openai/models")
    assert response.status_code == 200

    actual_models = [model["id"] for model in response.json()["data"]]
    expected_models = [option.value for option in BedrockModels]

    assert set(expected_models).issubset(
        set(actual_models)
    ), f"Expected models: {expected_models}, Actual models: {actual_models}"
