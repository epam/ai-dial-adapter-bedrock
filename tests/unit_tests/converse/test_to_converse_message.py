from aidial_sdk.chat_completion import FunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role as DialRole
from aidial_sdk.chat_completion import ToolCall

from aidial_adapter_bedrock.llm.converse.input import to_converse_message


async def test_to_converse_message_text():
    dial_message = DialMessage(role=DialRole.USER, content="Hello, world!")
    converse_message = await to_converse_message(dial_message, storage=None)

    assert converse_message == {
        "role": "user",
        "content": [{"text": "Hello, world!"}],
    }


async def test_to_converse_message_assistant():
    dial_message = DialMessage(role=DialRole.ASSISTANT, content="Hello")
    converse_message = await to_converse_message(dial_message, storage=None)

    assert converse_message == {
        "role": "assistant",
        "content": [{"text": "Hello"}],
    }


async def test_to_converse_message_function_call_no_content():
    dial_message = DialMessage(
        role=DialRole.ASSISTANT,
        function_call=FunctionCall(
            name="get_weather", arguments='{"city": "Paris"}'
        ),
    )
    converse_message = await to_converse_message(dial_message, storage=None)
    assert converse_message == {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "get_weather",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            },
        ],
    }


async def test_to_converse_message_function_call_with_content():
    dial_message = DialMessage(
        role=DialRole.ASSISTANT,
        content="Calling a function",
        function_call=FunctionCall(
            name="get_weather", arguments='{"city": "Paris"}'
        ),
    )
    converse_message = await to_converse_message(dial_message, storage=None)
    assert converse_message == {
        "role": "assistant",
        "content": [
            {"text": "Calling a function"},
            {
                "toolUse": {
                    "toolUseId": "get_weather",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            },
        ],
    }


async def test_to_converse_message_tool_call_no_content():
    dial_message = DialMessage(
        role=DialRole.ASSISTANT,
        tool_calls=[
            ToolCall(
                index=None,
                id="123",
                type="function",
                function=FunctionCall(
                    name="get_weather", arguments='{"city": "Paris"}'
                ),
            )
        ],
    )
    converse_message = await to_converse_message(dial_message, storage=None)
    assert converse_message == {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "123",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            },
        ],
    }


async def test_to_converse_message_tool_call_with_content():
    dial_message = DialMessage(
        role=DialRole.ASSISTANT,
        content="Calling a function",
        tool_calls=[
            ToolCall(
                index=None,
                id="123",
                type="function",
                function=FunctionCall(
                    name="get_weather", arguments='{"city": "Paris"}'
                ),
            )
        ],
    )
    converse_message = await to_converse_message(dial_message, storage=None)
    assert converse_message == {
        "role": "assistant",
        "content": [
            {"text": "Calling a function"},
            {
                "toolUse": {
                    "toolUseId": "123",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            },
        ],
    }


async def test_to_converse_message_multiple_tool_calls():
    dial_message = DialMessage(
        role=DialRole.ASSISTANT,
        tool_calls=[
            ToolCall(
                index=None,
                id="123",
                type="function",
                function=FunctionCall(
                    name="get_weather", arguments='{"city": "Paris"}'
                ),
            ),
            ToolCall(
                index=None,
                id="456",
                type="function",
                function=FunctionCall(
                    name="get_weather", arguments='{"city": "London"}'
                ),
            ),
        ],
    )
    converse_message = await to_converse_message(dial_message, storage=None)
    assert converse_message == {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "123",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            },
            {
                "toolUse": {
                    "toolUseId": "456",
                    "name": "get_weather",
                    "input": {"city": "London"},
                }
            },
        ],
    }
