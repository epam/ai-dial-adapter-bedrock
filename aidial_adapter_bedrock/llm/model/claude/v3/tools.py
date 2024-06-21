import json
from typing import assert_never

from aidial_sdk.chat_completion import FunctionCall, ToolCall
from anthropic.types import ToolUseBlock

from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
    BaseMessage,
    HumanFunctionResultMessage,
    HumanToolResultMessage,
    ToolMessage,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode


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


def process_tools_block(
    consumer: Consumer, block: ToolUseBlock, tools_mode: ToolsMode | None
):
    match tools_mode:
        case ToolsMode.TOOLS:
            consumer.create_function_tool_call(to_dial_tool_call(block))
        case ToolsMode.FUNCTIONS:
            consumer.create_function_call(to_dial_function_call(block))
        case None:
            raise ValidationError(
                "A model has called a tool, but no tools were given to the model in the first place."
            )
        case _:
            raise Exception(f"Unknown {tools_mode} during tool use!")


def convert_function_message(
    message: BaseMessage | ToolMessage,
) -> BaseMessage | ToolMessage:
    """
    If users uses functions instead of tools,
    for model we still need to convert function calls and result messages into tool calls

    For tool id we just use function name
    """
    if isinstance(message, BaseMessage):
        return message
    elif isinstance(message, HumanToolResultMessage) or isinstance(
        message, AIToolCallMessage
    ):
        raise ValidationError(
            "Tool messages are not allowed while using functions."
        )
    elif isinstance(message, HumanFunctionResultMessage):
        return HumanToolResultMessage(id=message.name, content=message.content)
    elif isinstance(message, AIFunctionCallMessage):
        return AIToolCallMessage(
            content=message.content,
            calls=[
                ToolCall(
                    index=None,
                    id=message.call.name,
                    type="function",
                    function=message.call,
                )
            ],
        )
    else:
        assert_never(message)
