from typing import List, Mapping

import pytest
from openai.types.chat import ChatCompletionMessageParam

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.llm.model.claude.v3.converters import MessageState
from tests.utils.openai import chat_completion, user

_EAST = "us-east-1"

chat_deployments: Mapping[ChatCompletionDeployment, str] = {
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET_US: _EAST,
}


@pytest.mark.parametrize(
    "deployment, region, stream",
    [
        (deployment, region, stream)
        for deployment, region in chat_deployments.items()
        for stream in [False, True]
    ],
)
async def test_claude_non_redacted_thinking(
    get_openai_client,
    deployment: ChatCompletionDeployment,
    region: str,
    stream: bool,
):
    client = get_openai_client(deployment.value, region=region)

    messages: List[ChatCompletionMessageParam] = [user("2+3=?")]

    configuration: dict = {
        "custom_fields": {
            "configuration": {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024,
                }
            }
        }
    }

    response1 = await chat_completion(
        client, messages=messages, stream=stream, extra_body=configuration
    )

    bot_message1 = response1.response.choices[0].message
    state_dict1 = bot_message1.custom_content["state"]  # type: ignore
    state1 = MessageState.parse_obj(state_dict1)
    assert len(state1.claude_message.content) == 2

    messages.append(bot_message1.dict())  # type: ignore
    messages.append(user("5+5=?"))

    response2 = await chat_completion(
        client, messages=messages, stream=stream, extra_body=configuration
    )

    bot_message2 = response2.response.choices[0].message
    state_dict2 = bot_message2.custom_content["state"]  # type: ignore
    state2 = MessageState.parse_obj(state_dict2)
    assert len(state2.claude_message.content) == 2


# NOTE: according to the Anthropic docs,
# https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#understanding-thinking-blocks
# a certain magic string is supposed to provoke a redacted thinking block.
# However, it doesn't for the Bedrock.
# Moreover, I didn't find a way to provoke a redacted thinking block in the Bedrock.
