from typing import List, Optional, Set

import pytest

from aidial_adapter_bedrock.llm.chat_model import default_keep_message
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.llm.model.llama_chat import (
    llama_emulator,
    llama_partitioner,
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
        keep_message=default_keep_message,
        partition_messages=llama_partitioner,
        model_limit=model_limit,
        user_limit=user_limit,
    )


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


multi_turn_dialogue = [
    # System
    SystemMessage(content="system"),
    # Turn 1
    HumanMessage(content="hello"),
    AIMessage(content="hi"),
    # Turn 2
    HumanMessage(content="ping"),
    AIMessage(content="pong"),
    # Final prompt
    HumanMessage(content="improvise"),
]


@pytest.mark.parametrize(
    "user_limit, expected",
    [
        (
            1,
            "Token count of the last message and all system messages (2) "
            "exceeds the maximum prompt tokens (1).",
        ),
        (2, {1, 2, 3, 4}),
        (3, {1, 2, 3, 4}),
        (4, {1, 2}),
        (5, {1, 2}),
        (6, set()),
    ],
)
def test_multi_turn_dialogue(user_limit: int, expected: Set[int] | str):
    discarded_messages = truncate_prompt_by_words(
        messages=multi_turn_dialogue, user_limit=user_limit
    )

    if isinstance(expected, str):
        assert (
            isinstance(discarded_messages, TruncatePromptError)
            and discarded_messages.print() == expected
        )
    else:
        assert discarded_messages == expected
