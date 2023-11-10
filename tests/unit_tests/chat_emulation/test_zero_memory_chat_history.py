from aidial_adapter_bedrock.llm.chat_emulation.history import FormattedMessage
from aidial_adapter_bedrock.llm.chat_emulation.zero_memory_chat import (
    ZeroMemoryChatHistory,
)
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
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
    history = ZeroMemoryChatHistory.create(messages)

    assert history.discarded_messages == 3
    assert history.messages == [
        FormattedMessage(text=" human message2 ", source_message=messages[3]),
    ]


def test_formatting():
    messages = [FormattedMessage(text="text")]
    history = ZeroMemoryChatHistory(messages=messages, discarded_messages=0)

    prompt = history.format()

    assert prompt == "text"
