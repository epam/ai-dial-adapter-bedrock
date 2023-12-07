from typing import List, Set

from aidial_adapter_bedrock.llm.chat_emulation.chat_emulator import (
    ChatEmulator,
    RolePrefixes,
    default_conf,
)
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
from aidial_adapter_bedrock.utils.list import omit_by_indices

noop_conf = ChatEmulator(
    prelude_template=None,
    add_role_prefix=lambda *_: False,
    add_invitation=False,
    fallback_to_completion=False,
    role_prefixes=RolePrefixes(system=None, human=None, ai=None),
    separator="",
)


def truncate_prompt_by_words(
    messages: List[BaseMessage], word_limit: int
) -> Set[int] | TruncatePromptError:
    def _count_words(messages: List[BaseMessage]) -> int:
        return sum(len(msg.content.split()) for msg in messages)

    return truncate_prompt(
        messages=messages,
        count_tokens=_count_words,
        keep_message=is_important_message,
        model_limit=None,
        user_limit=word_limit,
    )


def test_construction():
    messages = [
        SystemMessage(content=" system message1 "),
        HumanMessage(content="  human message1  "),
        AIMessage(content="     ai message1     "),
        HumanMessage(content="  human message2  "),
    ]

    text, stop_sequences = default_conf.display(messages)

    prelude = default_conf._prelude
    assert prelude is not None
    assert stop_sequences == ["\n\nHuman:"]
    assert text == "".join(
        [
            prelude,
            "\n\nHuman: system message1",
            "\n\nHuman: human message1",
            "\n\nAssistant: ai message1",
            "\n\nHuman: human message2",
            "\n\nAssistant:",
        ]
    )


def test_construction_with_single_user_message():
    messages: List[BaseMessage] = [HumanMessage(content=" human message ")]
    text, stop_sequences = default_conf.display(messages)

    assert stop_sequences == []
    assert text == " human message "


def test_formatting():
    messages = [
        SystemMessage(content="text1"),
        HumanMessage(content="text2"),
        AIMessage(content="text3"),
    ]

    text, stop_sequences = noop_conf.display(messages)

    assert stop_sequences == []
    assert text == "text1text2text3"


def test_no_trimming():
    messages = [
        SystemMessage(content="text1"),
        HumanMessage(content="text2"),
        AIMessage(content="text3"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, word_limit=3
    )

    assert discarded_messages == set()


def test_trimming():
    messages = [
        SystemMessage(content="system_message1"),
        HumanMessage(content="to_remove1"),
        SystemMessage(content="system_message2"),
        HumanMessage(content="to_remove2"),
        HumanMessage(content="query"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, word_limit=3
    )

    assert isinstance(discarded_messages, set)
    assert discarded_messages == {1, 3}

    text, stop_sequences = default_conf.display(
        omit_by_indices(messages, discarded_messages)
    )

    prelude = default_conf._prelude

    assert prelude is not None
    assert stop_sequences == ["\n\nHuman:"]
    assert text == "".join(
        [
            prelude,
            "\n\nHuman: system_message1",
            "\n\nHuman: system_message2",
            "\n\nHuman: query",
            "\n\nAssistant:",
        ]
    )


def test_trimming_with_one_message_left():
    messages = [
        AIMessage(content="reply"),
        HumanMessage(content="query"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, word_limit=1
    )

    assert isinstance(discarded_messages, set) and discarded_messages == {0}

    text, stop_sequences = default_conf.display(
        omit_by_indices(messages, discarded_messages)
    )

    assert stop_sequences == []
    assert text == "query"


def test_trimming_with_one_message_accepted_after_second_check():
    messages = [
        AIMessage(content="reply reply"),
        HumanMessage(content="query"),
    ]

    discarded_messages = truncate_prompt_by_words(
        messages=messages, word_limit=1
    )

    assert isinstance(discarded_messages, set)
    assert discarded_messages == {0}

    text, stop_sequences = default_conf.display(
        omit_by_indices(messages, discarded_messages)
    )

    assert stop_sequences == []
    assert text == "query"


def test_prompt_is_too_big():
    messages = [
        SystemMessage(content="text1"),
        SystemMessage(content="text2"),
        HumanMessage(content="text3"),
    ]

    truncation_error = truncate_prompt_by_words(messages=messages, word_limit=2)

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

    truncation_error = truncate_prompt_by_words(messages=messages, word_limit=1)

    assert (
        isinstance(truncation_error, TruncatePromptError)
        and truncation_error.print()
        == "Token count of the last message and all system messages (2) exceeds the maximum prompt tokens (1)."
    )
