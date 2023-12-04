from typing import List

import pytest

from aidial_adapter_bedrock.llm.chat_emulation.history import FormattedMessage
from aidial_adapter_bedrock.llm.chat_emulation.pseudo_chat import (
    PseudoChatHistory,
    default_conf,
)
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


def test_construction():
    messages = [
        SystemMessage(content="system message1"),
        HumanMessage(content=" human message1 "),
        AIMessage(content="    ai message1    "),
        HumanMessage(content=" human message2 "),
    ]
    history = PseudoChatHistory.create(messages, conf=default_conf)

    prelude = history.pseudo_history_conf.prelude
    assert prelude is not None
    assert history.stop_sequences == ["\n\nHuman:"]
    assert history.messages == [
        FormattedMessage(text=prelude),
        FormattedMessage(
            text="\n\nHuman: system message1", source_message=messages[0]
        ),
        FormattedMessage(
            text="\n\nHuman: human message1",
            source_message=messages[1],
            is_important=False,
        ),
        FormattedMessage(
            text="\n\nAssistant: ai message1",
            source_message=messages[2],
            is_important=False,
        ),
        FormattedMessage(
            text="\n\nHuman: human message2", source_message=messages[3]
        ),
        FormattedMessage(text="\n\nAssistant:"),
    ]


def test_construction_with_single_user_message():
    messages: List[BaseMessage] = [HumanMessage(content=" human message ")]
    history = PseudoChatHistory.create(messages, conf=default_conf)

    assert history.stop_sequences == []
    assert history.messages == [
        FormattedMessage(text=" human message ", source_message=messages[0])
    ]


def test_formatting():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2"),
        FormattedMessage(text="text3"),
    ]
    history = PseudoChatHistory(
        messages=messages, stop_sequences=[], pseudo_history_conf=default_conf
    )

    prompt = history.format()

    assert prompt == "text1text2text3"


def test_no_trimming():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2"),
        FormattedMessage(text="text3"),
    ]
    history = PseudoChatHistory(
        messages=messages, stop_sequences=[], pseudo_history_conf=default_conf
    )

    trimmed_history, discarded_messages_count = history.trim(lambda _: 1, 3)

    assert discarded_messages_count == 0
    assert trimmed_history == history


def test_trimming():
    messages = [
        FormattedMessage(
            text="\n\nHuman: system message1",
            source_message=SystemMessage(content="system message1"),
        ),
        FormattedMessage(text="to_remove1", is_important=False),
        FormattedMessage(
            text="\n\nHuman: system message2",
            source_message=SystemMessage(content="system message2"),
        ),
        FormattedMessage(text="to_remove2", is_important=False),
        FormattedMessage(
            text="\n\nHuman: query1",
            source_message=HumanMessage(content="query1"),
        ),
        FormattedMessage(text="\n\nAssistant:"),
    ]
    history = PseudoChatHistory(
        messages=messages, stop_sequences=[], pseudo_history_conf=default_conf
    )

    trimmed_history, discarded_messages_count = history.trim(lambda _: 1, 4)

    prelude = history.pseudo_history_conf.prelude
    assert prelude is not None
    assert discarded_messages_count == 2
    assert trimmed_history.stop_sequences == ["\n\nHuman:"]
    assert trimmed_history.messages == [
        FormattedMessage(text=prelude),
        FormattedMessage(
            text="\n\nHuman: system message1",
            source_message=SystemMessage(content="system message1"),
        ),
        FormattedMessage(
            text="\n\nHuman: system message2",
            source_message=SystemMessage(content="system message2"),
        ),
        FormattedMessage(
            text="\n\nHuman: query1",
            source_message=HumanMessage(content="query1"),
        ),
        FormattedMessage(text="\n\nAssistant:", source_message=None),
    ]


def test_trimming_with_one_message_left():
    messages = [
        FormattedMessage(
            text="text1",
            source_message=AIMessage(content="reply1"),
            is_important=False,
        ),
        FormattedMessage(
            text="text2",
            source_message=HumanMessage(content="query2"),
        ),
    ]
    history = PseudoChatHistory(
        messages=messages, stop_sequences=[], pseudo_history_conf=default_conf
    )

    trimmed_history, discarded_messages_count = history.trim(lambda _: 1, 1)

    assert discarded_messages_count == 1
    assert trimmed_history.stop_sequences == []
    assert trimmed_history.messages == [
        FormattedMessage(
            text="query2",
            source_message=HumanMessage(content="query2"),
        )
    ]


def test_trimming_with_one_message_accepted_after_second_check():
    messages = [
        FormattedMessage(
            text="text1",
            source_message=AIMessage(content="reply1"),
            is_important=False,
        ),
        FormattedMessage(
            text="text2",
            source_message=HumanMessage(content="query1"),
        ),
    ]
    history = PseudoChatHistory(
        messages=messages, stop_sequences=[], pseudo_history_conf=default_conf
    )

    trimmed_history, discarded_messages_count = history.trim(
        lambda text: 1 if text == "query1" else 2, 1
    )

    assert discarded_messages_count == 1
    assert trimmed_history.messages == [
        FormattedMessage(
            text="query1",
            source_message=HumanMessage(content="query1"),
        )
    ]


def test_prompt_is_too_big():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2"),
        FormattedMessage(text="text3"),
    ]
    history = PseudoChatHistory(
        messages=messages, stop_sequences=[], pseudo_history_conf=default_conf
    )

    with pytest.raises(ValidationError) as exc_info:
        history.trim(lambda _: 1, 2)

    assert (
        str(exc_info.value)
        == "Prompt token size (3) exceeds prompt token limit (2)."
    )


def test_prompt_with_history_is_too_big():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2", is_important=False),
        FormattedMessage(text="text3"),
    ]
    history = PseudoChatHistory(
        messages=messages, stop_sequences=[], pseudo_history_conf=default_conf
    )

    with pytest.raises(ValidationError) as exc_info:
        history.trim(lambda _: 1, 1)

    assert (
        str(exc_info.value)
        == "The token size of system messages and the last user message (2) exceeds prompt token limit (1)."
    )
