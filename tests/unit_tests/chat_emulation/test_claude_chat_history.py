import pytest

from aidial_adapter_bedrock.llm.chat_emulation.claude_chat import (
    ClaudeChatHistory,
)
from aidial_adapter_bedrock.llm.chat_emulation.history import FormattedMessage
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


@pytest.mark.parametrize("system_message_is_supported", [False, True])
def test_construction(system_message_is_supported: bool):
    messages = [
        SystemMessage(content=" system message1 "),
        HumanMessage(content="  human message1  "),
        AIMessage(content="     ai message1     "),
        HumanMessage(content="  human message2  "),
    ]
    history = ClaudeChatHistory.create(
        messages, system_message_is_supported=system_message_is_supported
    )

    sys_message_prefix = (
        "\n\nHuman: " if not system_message_is_supported else ""
    )

    assert history.messages == [
        FormattedMessage(
            text=f"{sys_message_prefix}system message1",
            source_message=messages[0],
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


def test_formatting():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2"),
        FormattedMessage(text="text3"),
    ]
    history = ClaudeChatHistory(messages=messages)

    prompt = history.format()

    assert prompt == "text1text2text3"


def test_no_trimming():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2"),
        FormattedMessage(text="text3"),
    ]
    history = ClaudeChatHistory(messages=messages)

    trimmed_history, discarded_messages_count = history.trim(lambda _: 1, 3)

    assert discarded_messages_count == 0
    assert trimmed_history == history


def test_trimming():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2", is_important=False),
        FormattedMessage(text="text3"),
        FormattedMessage(text="text4", is_important=False),
        FormattedMessage(text="text5"),
    ]
    history = ClaudeChatHistory(messages=messages)

    trimmed_history, discarded_messages_count = history.trim(lambda _: 1, 3)

    assert discarded_messages_count == 2
    assert trimmed_history.messages == [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text3"),
        FormattedMessage(text="text5"),
    ]


def test_prompt_is_too_big():
    messages = [
        FormattedMessage(text="text1"),
        FormattedMessage(text="text2"),
        FormattedMessage(text="text3"),
    ]
    history = ClaudeChatHistory(messages=messages)

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
    history = ClaudeChatHistory(messages=messages)

    with pytest.raises(ValidationError) as exc_info:
        history.trim(lambda _: 1, 1)

    assert (
        str(exc_info.value)
        == "The token size of system messages and the last user message (2) exceeds prompt token limit (1)."
    )
