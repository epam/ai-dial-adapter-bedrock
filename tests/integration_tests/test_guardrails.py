from dataclasses import dataclass
from typing import List, Mapping

import pytest
from openai.types.chat import ChatCompletionMessageParam

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from tests.integration_tests.test_chat_completion import Deployment
from tests.utils.openai import chat_completion, sanitize_test_name, user

_REGION = "us-west-2"

chat_deployments: Mapping[Deployment, str] = {
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET.US: _REGION,
}


_CONFIGURATION = {
    "custom_fields": {
        "configuration": {
            "guardrailConfig": {
                "guardrailIdentifier": "arn:aws:bedrock:us-west-2:725751206603:guardrail/hjiws0gnh52k",
                "guardrailVersion": "1",
                "trace": "enabled_full",
            }
        }
    }
}


@dataclass
class TestCase:
    __test__ = False

    deployment: Deployment
    region: str
    stream: bool

    def get_id(self) -> str:
        stream = "stream" if self.stream else "block"
        return sanitize_test_name(f"{stream}/{self.deployment.value}")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(deployment, region, stream)
        for deployment, region in chat_deployments.items()
        for stream in [False, True]
    ],
    ids=lambda t: t.get_id(),
)
async def test_claude_with_guardrails(get_openai_client, test_case: TestCase):
    stream = test_case.stream
    client = get_openai_client(
        test_case.deployment.value, region=test_case.region
    )

    messages: List[ChatCompletionMessageParam] = [
        user("Create a playlist of heavy metal songs")
    ]

    response = await chat_completion(
        client, messages=messages, stream=stream, extra_body=_CONFIGURATION
    )

    assert response.content == "Sorry, the model cannot answer this question."
    assert response.finish_reasons == ["content_filter"]
