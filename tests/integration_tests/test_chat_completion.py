import re
from dataclasses import dataclass
from typing import Callable, List

import openai
import openai.error
import pytest
from langchain.schema import BaseMessage

from aidial_adapter_bedrock.llm.bedrock_models import BedrockDeployment
from client.client_adapter import create_model
from tests.conftest import TEST_SERVER_URL
from tests.utils.llm import run_model, sanitize_test_name, sys, user


@dataclass
class TestCase:
    __test__ = False

    name: str
    deployment: BedrockDeployment
    streaming: bool

    messages: List[BaseMessage]
    test: Callable[[str], bool] | Exception

    def get_id(self):
        return sanitize_test_name(
            f"{self.deployment.value} {self.streaming} {self.name}"
        )


chat_deployments = [
    BedrockDeployment.AMAZON_TITAN_TG1_LARGE,
    BedrockDeployment.AI21_J2_GRANDE_INSTRUCT,
    BedrockDeployment.AI21_J2_JUMBO_INSTRUCT,
    BedrockDeployment.AI21_J2_MID,
    BedrockDeployment.AI21_J2_ULTRA,
    BedrockDeployment.ANTHROPIC_CLAUDE_INSTANT_V1,
    BedrockDeployment.ANTHROPIC_CLAUDE_V1,
    BedrockDeployment.ANTHROPIC_CLAUDE_V2,
]


def get_test_cases(
    deployment: BedrockDeployment, streaming: bool
) -> List[TestCase]:
    ret: List[TestCase] = []

    ret.append(
        TestCase(
            name="2+3=5",
            deployment=deployment,
            streaming=streaming,
            messages=[user("2+3=?")],
            test=lambda s: "5" in s,
        )
    )

    ret.append(
        TestCase(
            name="empty system message",
            deployment=deployment,
            streaming=streaming,
            messages=[sys(""), user("2+4=?")],
            test=lambda s: "6" in s,
        )
    )

    query = 'Reply with "Hello"'
    if deployment == BedrockDeployment.ANTHROPIC_CLAUDE_INSTANT_V1:
        query = 'Print "Hello"'

    ret.append(
        TestCase(
            name="hello",
            deployment=deployment,
            streaming=streaming,
            messages=[user(query)],
            test=lambda s: "hello" in s.lower(),
        )
    )

    ret.append(
        TestCase(
            name="empty dialog",
            deployment=deployment,
            streaming=streaming,
            messages=[],
            test=Exception("List of messages must not be empty"),
        )
    )

    ret.append(
        TestCase(
            name="empty user message",
            deployment=deployment,
            streaming=streaming,
            messages=[user("")],
            test=lambda s: True,
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test
        for deployment in chat_deployments
        for streaming in [False, True]
        for test in get_test_cases(deployment, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_langchain(server, test: TestCase):
    model = create_model(TEST_SERVER_URL, test.deployment.value, test.streaming)

    if isinstance(test.test, Exception):
        with pytest.raises(Exception) as exc_info:
            await run_model(model, test.messages, test.streaming)

        assert isinstance(exc_info.value, openai.error.OpenAIError)
        assert exc_info.value.http_status == 422
        assert re.search(str(test.test), str(exc_info.value))
    else:
        actual_output = await run_model(model, test.messages, test.streaming)
        assert test.test(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
