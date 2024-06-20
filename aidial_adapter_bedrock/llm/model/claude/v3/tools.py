from enum import Enum
from typing import assert_never
from aidial_sdk.chat_completion import FunctionCall, ToolCall
from anthropic.types import ToolUseBlock


import json

from aidial_adapter_bedrock.llm.consumer import Consumer


def to_dial_tool_call(block: ToolUseBlock) -> ToolCall:
    return ToolCall(
        index=None,
        id=block.id,
        type="function",
        function=FunctionCall(
            name=block.name,
            arguments=json.dumps(block.input),
        ),
    )


def to_dial_function_call(block: ToolUseBlock) -> FunctionCall:
    return FunctionCall(name=block.name, arguments=json.dumps(block.input))


class ToolsMode(Enum):
    NATIVE_TOOLS = "NATIVE_TOOLS"
    """
    Claude V3 API Supports only tools.
    But our API supports also deprecated "functions" which are pretty same as tools,
    so we can emulate them.
    """
    FUNCTION_EMULATION = "FUNCTION_EMULATION"


def process_tools_block(
    consumer: Consumer, block: ToolUseBlock, tools_mode: ToolsMode
):
    match tools_mode:
        case ToolsMode.NATIVE_TOOLS:
            consumer.create_function_tool_call(to_dial_tool_call(block))
        case ToolsMode.FUNCTION_EMULATION:
            consumer.create_function_call(to_dial_function_call(block))
        case _:
            assert_never(tools_mode)
