import httpx
import pytest

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.utils.openai import configuration

# Supported models and regions as per doc: https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html
deployments_supporting_optimized_latency: dict[
    ChatCompletionDeployment, list[str]
] = {
    ChatCompletionDeployment.AMAZON_NOVA_PRO: ["us-east-1", "us-east-2"],
    ChatCompletionDeployment.META_LLAMA3_1_70B_INSTRUCT_V1: [
        "us-east-2",
        "us-west-2",
    ],
    # Claude 3 only supports the optimize latency via the Converse API.
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU: [
        "us-east-2",
        "us-west-2",
    ],
}


async def _supports_optimized_latency(
    test_http_client: httpx.AsyncClient, deployment: ChatCompletionDeployment
) -> bool:
    conf = await configuration(test_http_client, deployment.value)
    assert conf is not None
    return "performanceConfig" in conf["properties"]


@pytest.mark.parametrize(
    "test", deployments_supporting_optimized_latency.items()
)
async def test_support_optimized_latency(
    test_http_client: httpx.AsyncClient,
    test: tuple[ChatCompletionDeployment, bool],
):
    deployment, regions = test

    expected_supported = regions != []
    if deployment == ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU:
        # Claude 3.5 supports optimized latency via fallback to Converse API.
        # Therefore, it's not declared in the model configuration schema.
        expected_supported = False

    actual_supported = await _supports_optimized_latency(
        test_http_client, deployment
    )
    assert expected_supported == actual_supported
