from openai import AsyncAzureOpenAI

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)


async def models_request_openai(client: AsyncAzureOpenAI) -> list[str]:
    data = (await client.models.list()).data
    return [model.id for model in data]


def assert_models_subset(actual_models: list[str]):
    def model_names(cls) -> list[str]:
        return [e.value for e in cls]

    expected_models = model_names(ChatCompletionDeployment) + model_names(
        EmbeddingsDeployment
    )

    assert set(expected_models).issubset(set(actual_models)), (
        f"Expected models: {expected_models}, Actual models: {actual_models}"
    )


async def test_model_list_openai(get_openai_client):
    assert_models_subset(await models_request_openai(get_openai_client()))
