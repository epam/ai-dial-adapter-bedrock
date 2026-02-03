from dataclasses import dataclass
from typing import List, Mapping

import pytest
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.llm.model.claude.v3.state import MessageState
from tests.integration_tests.test_chat_completion import Deployment
from tests.utils.openai import (
    GET_WEATHER_FUNCTION,
    chat_completion,
    function_to_tool,
    sanitize_test_name,
    user,
)

_EAST = "us-east-1"

chat_deployments: Mapping[Deployment, str] = {
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET.US: _EAST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_OPUS.US: _EAST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_1_OPUS.US: _EAST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_SONNET.US: _EAST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_5_HAIKU.US: _EAST,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_5_SONNET.US: _EAST,
}


def supports_parallel_tool_calls(deployment: ChatCompletionDeployment):
    return deployment != ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET


_CONFIGURATION = {
    "custom_fields": {
        "configuration": {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024,
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
async def test_claude_thinking_no_function_calling(
    get_openai_client, test_case: TestCase
):
    stream = test_case.stream
    client = get_openai_client(
        test_case.deployment.value, region=test_case.region
    )

    messages: List[ChatCompletionMessageParam] = [user("2+3=?")]

    response1 = await chat_completion(
        client, messages=messages, stream=stream, extra_body=_CONFIGURATION
    )

    bot_message1 = response1.response.choices[0].message
    state_dict1 = bot_message1.custom_content["state"]  # type: ignore
    state1 = MessageState.model_validate(state_dict1)
    assert len(state1.claude_message_content) == 2

    messages.append(bot_message1.model_dump())  # type: ignore
    messages.append(user("5+5=?"))

    response2 = await chat_completion(
        client, messages=messages, stream=stream, extra_body=_CONFIGURATION
    )

    bot_message2 = response2.response.choices[0].message
    state_dict2 = bot_message2.custom_content["state"]  # type: ignore
    state2 = MessageState.model_validate(state_dict2)
    assert len(state2.claude_message_content) == 2


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(deployment, region, stream)
        for deployment, region in chat_deployments.items()
        for stream in [False, True]
    ],
    ids=lambda t: t.get_id(),
)
async def test_claude_thinking_with_function_calling(
    get_openai_client, test_case: TestCase
):
    stream = test_case.stream
    client = get_openai_client(
        test_case.deployment.value, region=test_case.region
    )

    cities = ["Glasgow", "London"]
    temps = [10, 23]
    if not supports_parallel_tool_calls(test_case.deployment.origin):
        cities = cities[:1]
        temps = temps[:1]

    messages: List[ChatCompletionMessageParam] = [
        user(
            f"Tell me what's the temperature in {' and in '.join(cities)} in celsius?"
        )
    ]

    response1 = await chat_completion(
        client,
        messages=messages,
        stream=stream,
        tools=[function_to_tool(GET_WEATHER_FUNCTION)],
        extra_body=_CONFIGURATION,
    )

    bot_message1 = response1.response.choices[0].message
    state_dict1 = bot_message1.custom_content["state"]  # type: ignore
    state1 = MessageState.model_validate(state_dict1)
    assert len(state1.claude_message_content) > 0

    messages.append(bot_message1.model_dump())  # type: ignore

    tool_calls = bot_message1.tool_calls
    assert tool_calls is not None, "No tool calls were made"
    assert len(tool_calls) == len(cities)

    for tool_call, temp in zip(tool_calls, temps):
        messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=tool_call.id,
                content=f"{temp} degrees",
            )
        )

    response2 = await chat_completion(
        client,
        messages=messages,
        stream=stream,
        tools=[function_to_tool(GET_WEATHER_FUNCTION)],
        extra_body=_CONFIGURATION,
    )

    bot_message2 = response2.response.choices[0].message
    state_dict2 = bot_message2.custom_content["state"]  # type: ignore
    state2 = MessageState.model_validate(state_dict2)
    assert len(state2.claude_message_content) > 0

    for temp in temps:
        assert str(temp) in (bot_message2.content or "")


# NOTE: according to the Anthropic docs,
# https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#understanding-thinking-blocks
# a certain magic string is supposed to provoke a redacted thinking block.
# However, it doesn't for the Bedrock.
# Moreover, I didn't find a way to provoke a redacted thinking block in the Bedrock.
