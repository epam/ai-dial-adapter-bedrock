import math
from typing import List
from unittest.mock import patch

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
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from tests.utils.messages import ai, sys, to_sdk_messages, user, user_with_image

_DEPLOYMENT = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS

_MODEL = Claude_V3.create(_DEPLOYMENT, "-", AWSClientConfig(region="us-east-1"))


async def tokenize(
    messages: List[Message],
    tool_config: ToolsConfig | None = None,
) -> int:
    params = ModelParameters(tool_config=tool_config)
    return await _MODEL.count_prompt_tokens(params, messages)


async def compute_discarded_messages(
    messages: List[Message],
    max_prompt_tokens: int | None,
    tool_config: ToolsConfig | None = None,
) -> DiscardedMessages | str:
    params = ModelParameters(
        max_prompt_tokens=max_prompt_tokens, tool_config=tool_config
    )

    try:
        return await _MODEL.compute_discarded_messages(params, messages) or []
    except DialException as e:
        return e.message


_TOOL_CONFIG = ToolsConfig(
    functions=[Function(name="function")],
    required=False,
    tool_ids={},
)

_PER_MESSAGE_TOKENS = 5

_PNG_IMAGE_50_50 = "iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAIAAACRXR/mAAAAS0lEQVR4nO3OsQEAEADAMPz/Mw9YMjE0F2Tu8aP1OnBXS9QStUQtUUvUErVELVFL1BK1RC1RS9QStUQtUUvUErVELVFL1BK1RC1xAEGqAWOFuDKrAAAAAElFTkSuQmCC"

_PNG_IMAGE_50_50_TOKENS = math.ceil((50 * 50) / 750.0)


@pytest.fixture
def mock_tokenize_text():
    with patch(
        "aidial_adapter_bedrock.llm.model.claude.v3.tokenizer.tokenize_text"
    ) as mock:

        def _tokenize(txt: str):
            try:
                return int(txt)
            except Exception:
                return 1

        mock.side_effect = _tokenize
        yield mock


@pytest.mark.asyncio
async def test_one_turn_no_truncation(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("11"),
            user("22"),
            ai("33"),
        ]
    )

    expected_tokens = (
        11 + (_PER_MESSAGE_TOKENS + 22) + (_PER_MESSAGE_TOKENS + 33)
    )

    assert await tokenize(messages) == expected_tokens

    discarded_messages = await compute_discarded_messages(
        messages, expected_tokens
    )

    assert discarded_messages == []


@pytest.mark.asyncio
async def test_one_turn_with_image(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("11"),
            user_with_image("22", _PNG_IMAGE_50_50),
        ]
    )

    expected_tokens = 11 + (_PER_MESSAGE_TOKENS + _PNG_IMAGE_50_50_TOKENS + 22)

    assert await tokenize(messages) == expected_tokens

    truncation = await compute_discarded_messages(messages, expected_tokens)

    assert truncation == []

    truncation = await compute_discarded_messages(messages, expected_tokens - 1)

    assert (
        truncation
        == f"The requested maximum prompt tokens is {expected_tokens-1}. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_one_turn_with_tools(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("11"),
            user("22"),
            ai("33"),
        ]
    )

    expected_tokens = (
        530 + 11 + 1 + (_PER_MESSAGE_TOKENS + 22) + (_PER_MESSAGE_TOKENS + 33)
    )

    assert await tokenize(messages, _TOOL_CONFIG) == expected_tokens

    discarded_messages = await compute_discarded_messages(
        messages, expected_tokens - 1, _TOOL_CONFIG
    )

    assert (
        discarded_messages
        == f"The requested maximum prompt tokens is {expected_tokens-1}. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_one_turn_overflow(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("11"),
            user("22"),
            ai("33"),
        ]
    )

    expected_tokens = (
        11 + (22 + _PER_MESSAGE_TOKENS) + (33 + _PER_MESSAGE_TOKENS)
    )

    truncation_error = await compute_discarded_messages(messages, 1)

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_multiple_system_messages(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("system1"),
            sys("system2"),
            user("user"),
        ]
    )

    with pytest.raises(ValidationError) as exc_info:
        await compute_discarded_messages(messages, 3)

        assert exc_info.value.message == (
            "System message is only allowed as the first message"
        )


@pytest.mark.asyncio
async def test_truncate_first_turn(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            user("11"),
            ai("22"),
            user("33"),
            ai("44"),
        ]
    )

    expected_tokens = (
        (_PER_MESSAGE_TOKENS + 11)
        + (_PER_MESSAGE_TOKENS + 22)
        + (_PER_MESSAGE_TOKENS + 33)
        + (_PER_MESSAGE_TOKENS + 44)
    )

    assert await tokenize(messages) == expected_tokens

    discarded_messages = await compute_discarded_messages(
        messages, (_PER_MESSAGE_TOKENS + 33) + (_PER_MESSAGE_TOKENS + 44)
    )

    assert discarded_messages == [0, 1]


@pytest.mark.asyncio
async def test_truncate_first_turn_with_system(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("11"),
            user("22"),
            ai("33"),
            user("44"),
            ai("55"),
        ]
    )

    discarded_messages = await compute_discarded_messages(
        messages, 11 + (_PER_MESSAGE_TOKENS + 44) + (_PER_MESSAGE_TOKENS + 55)
    )

    assert discarded_messages == [1, 2]


@pytest.mark.asyncio
async def test_zero_turn_overflow(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("11"),
            user("22"),
        ]
    )

    expected_tokens = 11 + (22 + _PER_MESSAGE_TOKENS)

    truncation_error = await compute_discarded_messages(messages, 3)

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 3. However, the system messages and the last user message resulted in {expected_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


@pytest.mark.asyncio
async def test_chat_history_overflow(mock_tokenize_text):
    messages = to_sdk_messages(
        [
            sys("11"),
            user("22"),
            ai("33"),
            user("44"),
        ]
    )

    min_possible_tokens = 11 + (44 + _PER_MESSAGE_TOKENS)

    truncation_error = await compute_discarded_messages(messages, 1)

    assert (
        truncation_error
        == f"The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in {min_possible_tokens} tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )
