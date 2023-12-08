from typing import List

import pytest

from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.llm.model.llama_chat import llama_emulator


def test_construction_single_message():
    messages: List[BaseMessage] = [
        HumanMessage(content="  human message1  "),
    ]

    text, stop_sequences = llama_emulator.display(messages)

    assert stop_sequences == []
    assert text == "<s>[INST] human message1 [/INST]"


def test_construction_many_without_system():
    messages = [
        HumanMessage(content="  human message1  "),
        AIMessage(content="     ai message1     "),
        HumanMessage(content="  human message2  "),
    ]

    text, stop_sequences = llama_emulator.display(messages)

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
        SystemMessage(content=" system message1 "),
        HumanMessage(content="  human message1  "),
        AIMessage(content="     ai message1     "),
        HumanMessage(content="  human message2  "),
    ]

    text, stop_sequences = llama_emulator.display(messages)

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
        AIMessage(content="     ai message1     "),
        HumanMessage(content="  human message1  "),
        HumanMessage(content="  human message2  "),
    ]

    with pytest.raises(ValidationError) as exc_info:
        llama_emulator.display(messages)

    assert exc_info.value.message == (
        "The model only supports initial optional system message and"
        " follow-up alternating human/assistant messages"
    )


def test_invalid_last_message():
    messages = [
        HumanMessage(content="  human message1  "),
        AIMessage(content="     ai message1     "),
        HumanMessage(content="  human message2  "),
        AIMessage(content="     ai message2     "),
    ]

    with pytest.raises(ValidationError) as exc_info:
        llama_emulator.display(messages)

    assert exc_info.value.message == "The last message must be from user"
