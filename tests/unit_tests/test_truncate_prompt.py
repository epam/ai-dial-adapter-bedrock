from typing import List, Optional

from aidial_sdk.chat_completion import Message, Role

from aidial_adapter_bedrock.llm.chat_model import keep_last, trivial_partitioner
from aidial_adapter_bedrock.llm.truncate_prompt import (
    DiscardedMessages,
    TruncatePromptError,
    _partition_indexer,
    compute_discarded_messages,
)
from tests.utils.messages import ai, sys, user


def keep_last_and_system_messages(messages: List[Message], idx: int) -> bool:
    return messages[idx].role == Role.SYSTEM or keep_last(messages, idx)


async def truncate_prompt_by_words(
    messages: List[Message],
    user_limit: int,
    model_limit: Optional[int] = None,
) -> DiscardedMessages | TruncatePromptError:
    async def _tokenize_by_words(messages: List[Message]) -> int:
        return sum(len(msg.text().split()) for msg in messages)

    return await compute_discarded_messages(
        messages=messages,
        tokenizer=_tokenize_by_words,
        keep_message=keep_last_and_system_messages,
        partitioner=trivial_partitioner,
        model_limit=model_limit,
        user_limit=user_limit,
    )


def test_partition_indexer():
    assert [_partition_indexer([2, 3])(i) for i in range(5)] == [
        [0, 1],
        [0, 1],
        [2, 3, 4],
        [2, 3, 4],
        [2, 3, 4],
    ]


async def test_no_truncation():
    messages = [
        sys("text1"),
        user("text2"),
        ai("text3"),
    ]

    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=3
    )

    assert discarded_messages == []


async def test_truncation():
    messages = [
        sys("system1"),
        user("remove1"),
        sys("system2"),
        user("remove2"),
        user("query"),
    ]
    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=3
    )

    assert discarded_messages == [1, 3]


async def test_truncation_with_one_message_left():
    messages = [
        ai("reply"),
        user("query"),
    ]

    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert discarded_messages == [0]


async def test_truncation_with_one_message_accepted_after_second_check():
    messages = [
        ai("hello world"),
        user("query"),
    ]

    discarded_messages = await truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert discarded_messages == [0]


async def test_prompt_is_too_big():
    messages = [
        sys("text1"),
        sys("text2"),
        user("text3"),
    ]

    truncation_error = await truncate_prompt_by_words(
        messages=messages, user_limit=2
    )

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "The requested maximum prompt tokens is 2. However, the system messages and the last user message resulted in 3 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_prompt_with_history_is_too_big():
    messages = [
        sys("text1"),
        ai("text2"),
        user("text3"),
    ]

    truncation_error = await truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "The requested maximum prompt tokens is 1. However, the system messages and the last user message resulted in 2 tokens. Please reduce the length of the messages or increase the maximum prompt tokens."
    )


async def test_inconsistent_limits():
    messages = [ai("text2")]

    truncation_error = await truncate_prompt_by_words(
        messages=messages, user_limit=10, model_limit=5
    )

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "The request maximum prompt tokens is 10. However, the model's maximum context length is 5 tokens."
    )
