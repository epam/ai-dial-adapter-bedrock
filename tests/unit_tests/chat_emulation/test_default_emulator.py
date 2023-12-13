from typing import List

from aidial_adapter_bedrock.llm.chat_emulator import (
    BasicChatEmulator,
    CueMapping,
    default_emulator,
)
from aidial_adapter_bedrock.llm.message import BaseMessage
from tests.utils.messages import ai, sys, user

noop_emulator = BasicChatEmulator(
    prelude_template=None,
    add_cue=lambda *_: False,
    add_invitation_cue=False,
    fallback_to_completion=False,
    cues=CueMapping(system=None, human=None, ai=None),
    separator="",
)


def test_construction():
    messages = [
        sys(" system message1 "),
        user("  human message1  "),
        ai("     ai message1     "),
        user("  human message2  "),
    ]

    text, stop_sequences = default_emulator.display(messages)

    prelude = default_emulator._prelude
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
    messages: List[BaseMessage] = [user(" human message ")]
    text, stop_sequences = default_emulator.display(messages)

    assert stop_sequences == []
    assert text == " human message "


def test_construction_with_single_ai_message():
    messages: List[BaseMessage] = [ai(" ai message ")]
    text, stop_sequences = default_emulator.display(messages)

    prelude = default_emulator._prelude
    assert prelude is not None
    assert stop_sequences == ["\n\nHuman:"]
    assert text == "".join(
        [
            prelude,
            "\n\nAssistant: ai message",
            "\n\nAssistant:",
        ]
    )


def test_formatting():
    messages = [
        sys("text1"),
        user("text2"),
        ai("text3"),
    ]

    text, stop_sequences = noop_emulator.display(messages)

    assert stop_sequences == []
    assert text == "text1text2text3"
