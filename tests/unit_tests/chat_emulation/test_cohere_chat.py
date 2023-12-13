from typing import List

from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.cohere import cohere_emulator
from tests.utils.messages import ai, sys, user


def test_construction1():
    messages: List[BaseMessage] = [
        user("  human message1  "),
    ]

    text, stop_sequences = cohere_emulator.display(messages)

    assert stop_sequences == ["\nUser:"]
    assert text == "human message1"


def test_construction2():
    messages = [
        sys(" system message1 "),
        user("  human message1  "),
        ai("     ai message1     "),
        user("  human message2  "),
    ]

    text, stop_sequences = cohere_emulator.display(messages)

    assert stop_sequences == ["\nUser:"]
    assert text == "\n".join(
        [
            "system message1",
            "User: human message1",
            "Chatbot: ai message1",
            "User: human message2",
        ]
    )
