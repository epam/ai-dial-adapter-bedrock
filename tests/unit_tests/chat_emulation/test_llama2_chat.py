from typing import List, Optional, Set

import pytest

from aidial_adapter_bedrock.llm.chat_model import keep_last_and_system_messages
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.llama.v2 import llama2_config
from aidial_adapter_bedrock.llm.truncate_prompt import (
    TruncatePromptError,
    truncate_prompt_,
)
from tests.utils.messages import ai, sys, user

llama2_chat_emulator = llama2_config.chat_emulator
llama2_chat_partitioner = llama2_config.chat_partitioner


async def truncate_prompt_by_words(
    messages: List[BaseMessage],
    user_limit: int,
    model_limit: Optional[int] = None,
) -> Set[int] | TruncatePromptError:
    async def _tokenize_by_words(messages: List[BaseMessage]) -> int:
        return sum(len(msg.content.split()) for msg in messages)

    return await truncate_prompt_(
        messages=messages,
        tokenize_messages=_tokenize_by_words,
        keep_message=keep_last_and_system_messages,
        partition_messages=llama2_chat_partitioner,
        model_limit=model_limit,
        user_limit=user_limit,
    )


def test_construction_single_message():
    messages: List[BaseMessage] = [
        user("  human message1  "),
    ]

    text, stop_sequences = llama2_chat_emulator.display(messages)

    assert stop_sequences == []
    assert text == "<s>[INST] human message1 [/INST]"


def test_construction_many_without_system():
    messages = [
        user("  human message1  "),
        ai("     ai message1     "),
        user("  human message2  "),
    ]

    text, stop_sequences = llama2_chat_emulator.display(messages)

    assert stop_sequences == []
    assert text == "".join(
        [
            "<s>[INST] human message1 [/INST]",
            " ai message1 </s>",
            "<s>[INST] human message2 [/INST]",
        ]
    )


def test_construction_many_with_system():
    messages = [
        sys(" system message1 "),
        user("  human message1  "),
        ai("     ai message1     "),
        user("  human message2  "),
    ]

    text, stop_sequences = llama2_chat_emulator.display(messages)

    assert stop_sequences == []
    assert text == "".join(
        [
            "<s>[INST] <<SYS>>\n system message1 \n<</SYS>>\n\n  human message1 [/INST]",
            " ai message1 </s>",
            "<s>[INST] human message2 [/INST]",
        ]
    )


def test_invalid_alternation():
    messages = [
        ai("     ai message1     "),
        user("  human message1  "),
        user("  human message2  "),
    ]

    with pytest.raises(ValidationError) as exc_info:
        llama2_chat_emulator.display(messages)

    assert exc_info.value.message == (
        "The model only supports initial optional system message and"
        " follow-up alternating human/assistant messages"
    )


def test_invalid_last_message():
    messages = [
        user("  human message1  "),
        ai("     ai message1     "),
        user("  human message2  "),
        ai("     ai message2     "),
    ]

    with pytest.raises(ValidationError) as exc_info:
        llama2_chat_emulator.display(messages)

    assert exc_info.value.message == "The last message must be from user"


turns_sys = [
    sys("system"),
    user("hello"),
    ai("hi"),
    user("ping"),
    ai("pong"),
    user("improvise"),
]

turns_no_sys = turns_sys[1:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages, user_limit, expected",
    [
        (
            turns_sys,
            1,
            "The requested maximum prompt tokens is 1. "
            "However, the system messages and the last user message resulted in 2 tokens. "
            "Please reduce the length of the messages or increase the maximum prompt tokens.",
        ),
        (turns_sys, 2, {1, 2, 3, 4}),
        (turns_sys, 3, {1, 2, 3, 4}),
        (turns_sys, 4, {1, 2}),
        (turns_sys, 5, {1, 2}),
        (turns_sys, 6, set()),
        (turns_no_sys, 1, {0, 1, 2, 3}),
        (turns_no_sys, 2, {0, 1, 2, 3}),
        (turns_no_sys, 3, {0, 1}),
        (turns_no_sys, 4, {0, 1}),
        (turns_no_sys, 5, set()),
    ],
)
async def test_multi_turn_dialogue(
    messages: List[BaseMessage], user_limit: int, expected: Set[int] | str
):
    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=user_limit
    )

    if isinstance(expected, str):
        assert (
            isinstance(discarded_messages, TruncatePromptError)
            and discarded_messages.print() == expected
        )
    else:
        assert discarded_messages == expected
