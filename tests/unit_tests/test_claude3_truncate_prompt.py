from typing import List, Set

import pytest
from aidial_sdk.chat_completion import Function, Message
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.model.claude.v3.adapter import (
    Adapter as Claude_V3,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsConfig
from tests.utils.messages import ai, sys, to_sdk_messages, user


async def truncate_prompt(
    messages: List[Message],
    max_prompt_tokens: int | None,
    tool_config: ToolsConfig | None = None,
) -> Set[int] | str:
    deployment = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS
    model = Claude_V3.create(
        deployment, "-", AWSClientConfig(region="us-east-1")
    )
    params = ModelParameters(
        max_prompt_tokens=max_prompt_tokens,
        tool_config=tool_config,
    )

    try:
        return set(await model.truncate_prompt(params, messages) or [])
    except DialException as e:
        return e.message


_TOOL_CONFIG = ToolsConfig(
    functions=[Function(name="f")],
    required=False,
    tool_ids={},
)

_PER_MESSAGE_TOKENS = 5


@pytest.mark.asyncio
async def test_one_turn_no_truncation():
    messages = to_sdk_messages(
        [
            sys("s"),
            user("u"),
            ai("a"),
        ]
    )

    discarded_messages = await truncate_prompt(
        messages=messages, max_prompt_tokens=1 + 2 * (_PER_MESSAGE_TOKENS + 1)
    )

    assert discarded_messages == set()


@pytest.mark.asyncio
async def test_one_turn_with_tools():
    messages = to_sdk_messages(
        [
            sys("s"),
            user("u"),
            ai("a"),
        ]
    )

    discarded_messages = await truncate_prompt(
        messages=messages,
        max_prompt_tokens=530 + 85 + 1 + 2 * (_PER_MESSAGE_TOKENS + 1),
        tool_config=_TOOL_CONFIG,
    )

    assert (
        discarded_messages
        == "The requested maximum prompt tokens is 628. However, the system messages and the last user message resulted in 629 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_one_turn_overflow():
    messages = to_sdk_messages(
        [
            sys("s"),
            user("u"),
            ai("a"),
        ]
    )

    truncation_error = await truncate_prompt(
        messages=messages, max_prompt_tokens=1
    )

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in {1 + 2*(1 + _PER_MESSAGE_TOKENS)} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_multiple_system_messages():
    messages = to_sdk_messages(
        [
            sys("s"),
            sys("s"),
            user("r"),
        ]
    )

    with pytest.raises(ValidationError) as exc_info:
        await truncate_prompt(messages=messages, max_prompt_tokens=3)

        assert exc_info.value.message == (
            "System message is only allowed as the first message"
        )


@pytest.mark.asyncio
async def test_truncate_first_turn():
    messages = to_sdk_messages(
        [
            ai("a"),
            user("u"),
            ai("a"),
            user("u"),
        ]
    )

    discarded_messages = await truncate_prompt(
        messages=messages, max_prompt_tokens=2 * (_PER_MESSAGE_TOKENS + 1)
    )

    assert discarded_messages == {0, 1}


@pytest.mark.asyncio
async def test_truncate_first_turn_with_system():
    messages = to_sdk_messages(
        [
            sys("s"),
            user("u"),
            ai("a"),
            user("u"),
            ai("a"),
        ]
    )

    discarded_messages = await truncate_prompt(
        messages=messages, max_prompt_tokens=1 + 2 * (_PER_MESSAGE_TOKENS + 1)
    )

    assert discarded_messages == {1, 2}


@pytest.mark.asyncio
async def test_zero_turn_overflow():
    messages = to_sdk_messages(
        [
            sys("s"),
            user("u"),
        ]
    )

    truncation_error = await truncate_prompt(
        messages=messages, max_prompt_tokens=3
    )

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 3. However, the system messages and the last user message resulted in {1+(1 + _PER_MESSAGE_TOKENS)} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_chat_history_overflow():
    messages = to_sdk_messages(
        [
            sys("s"),
            user("u"),
            ai("a"),
            user("u"),
        ]
    )

    truncation_error = await truncate_prompt(
        messages=messages, max_prompt_tokens=1
    )

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in {1 + (1 + _PER_MESSAGE_TOKENS)} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )
