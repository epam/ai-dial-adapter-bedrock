from typing import List

from aidial_sdk.chat_completion import Message

from aidial_adapter_bedrock.llm.chat_emulator import (
    BasicChatEmulator,
    CueMapping,
    default_emulator,
)
from tests.utils.messages import ai, sys, to_sdk_messages, user

noop_emulator = BasicChatEmulator(
    prelude_template=None,
    add_cue=lambda *_: False,
    add_invitation_cue=False,
    fallback_to_completion=False,
    cues=CueMapping(system=None, human=None, ai=None),
    separator="",
)


def test_construction():
    messages: List[Message] = to_sdk_messages(
        [
            sys(" system message1 "),
            user("  human message1  "),
            ai("     ai message1     "),
            user("  human message2  "),
        ]
    )

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
    messages: List[Message] = [user(" human message ").to_message()]
    text, stop_sequences = default_emulator.display(messages)

    assert stop_sequences == []
    assert text == " human message "


def test_construction_with_single_ai_message():
    messages: List[Message] = [ai(" ai message ").to_message()]
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
    messages: List[Message] = [
        sys("text1").to_message(),
        user("text2").to_message(),
        ai("text3").to_message(),
    ]

    text, stop_sequences = noop_emulator.display(messages)

    assert stop_sequences == []
    assert text == "text1text2text3"
