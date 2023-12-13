from typing import List

from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    BaseMessage,
    HumanRegularMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.llm.model.cohere import cohere_emulator


def test_construction1():
    messages: List[BaseMessage] = [
        HumanRegularMessage(content="  human message1  "),
    ]

    text, stop_sequences = cohere_emulator.display(messages)

    assert stop_sequences == ["\nUser:"]
    assert text == "human message1"


def test_construction2():
    messages = [
        SystemMessage(content=" system message1 "),
        HumanRegularMessage(content="  human message1  "),
        AIRegularMessage(content="     ai message1     "),
        HumanRegularMessage(content="  human message2  "),
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
