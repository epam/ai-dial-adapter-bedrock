from typing import List, Optional, Set

from aidial_adapter_bedrock.llm.chat_model import is_important_message
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.llm.truncate_prompt import (
    TruncatePromptError,
    truncate_prompt,
)


def truncate_prompt_by_words(
    messages: List[BaseMessage],
    user_limit: int,
    model_limit: Optional[int] = None,
) -> Set[int] | TruncatePromptError:
    def _tokenize_by_words(messages: List[BaseMessage]) -> int:
        return sum(len(msg.content.split()) for msg in messages)

    return truncate_prompt(
        messages=messages,
        tokenize=_tokenize_by_words,
        keep_message=is_important_message,
        model_limit=model_limit,
        user_limit=user_limit,
    )


def test_no_truncation():
    messages = [
        SystemMessage(content="text1"),
        HumanMessage(content="text2"),
        AIMessage(content="text3"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, user_limit=3
    )

    assert discarded_messages == set()


def test_truncation():
    messages = [
        SystemMessage(content="system1"),
        HumanMessage(content="remove1"),
        SystemMessage(content="system2"),
        HumanMessage(content="remove2"),
        HumanMessage(content="query"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, user_limit=3
    )

    assert isinstance(discarded_messages, set)
    assert discarded_messages == {1, 3}


def test_truncation_with_one_message_left():
    messages = [
        AIMessage(content="reply"),
        HumanMessage(content="query"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert isinstance(discarded_messages, set) and discarded_messages == {0}


def test_truncation_with_one_message_accepted_after_second_check():
    messages = [
        AIMessage(content="hello world"),
        HumanMessage(content="query"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, user_limit=1
    )

    assert isinstance(discarded_messages, set)
    assert discarded_messages == {0}


def test_prompt_is_too_big():
    messages = [
        SystemMessage(content="text1"),
        SystemMessage(content="text2"),
        HumanMessage(content="text3"),
    ]

    truncation_error = truncate_prompt_by_words(messages=messages, user_limit=2)

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "Token count of the last message and all system messages (3) exceeds the maximum prompt tokens (2)."
    )


def test_prompt_with_history_is_too_big():
    messages = [
        SystemMessage(content="text1"),
        AIMessage(content="text2"),
        HumanMessage(content="text3"),
    ]

    truncation_error = truncate_prompt_by_words(messages=messages, user_limit=1)

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "Token count of the last message and all system messages (2) exceeds the maximum prompt tokens (1)."
    )


def test_inconsistent_limits():
    messages: List[BaseMessage] = [AIMessage(content="text2")]

    truncation_error = truncate_prompt_by_words(
        messages=messages, user_limit=10, model_limit=5
    )

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "Maximum prompt tokens (10) exceeds the model maximum prompt tokens (5)."
    )
