import json
from typing import assert_never

from aidial_sdk.chat_completion import FunctionCall, ToolCall
from anthropic.types.beta import BetaToolUseBlock as ToolUseBlock

from aidial_adapter_bedrock.llm.consumer import Consumer, ToolUseMessage
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIRegularMessage,
    AIToolCallMessage,
    BaseMessage,
    HumanFunctionResultMessage,
    HumanRegularMessage,
    HumanToolResultMessage,
    SystemMessage,
    ToolMessage,
)
from aidial_adapter_bedrock.llm.tools.tools_config import ToolsMode
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def to_dial_function_call(block: ToolUseBlock, streaming: bool) -> FunctionCall:
    arguments = "" if streaming else json.dumps(block.input)
    return FunctionCall(name=block.name, arguments=arguments)


def to_dial_tool_call(block: ToolUseBlock, streaming: bool) -> ToolCall:
    return ToolCall(
        index=None,
        id=block.id,
        type="function",
        function=to_dial_function_call(block, streaming),
    )


def process_tools_block(
    consumer: Consumer,
    block: ToolUseBlock,
    tools_mode: ToolsMode | None,
    *,
    streaming: bool,
) -> ToolUseMessage | None:
    match tools_mode:
        case ToolsMode.TOOLS:
            return consumer.create_function_tool_call(
                to_dial_tool_call(block, streaming)
            )
        case ToolsMode.FUNCTIONS:
            if consumer.has_function_call:
                log.warning(
                    "The model generated more than one tool call. "
                    "Only the first one will be taken in to account."
                )
                return None
            else:
                return consumer.create_function_call(
                    to_dial_function_call(block, streaming)
                )
        case None:
            raise ValidationError(
                "A model has called a tool, but no tools were given to the model in the first place."
            )
        case _:
            assert_never(tools_mode)


def function_to_tool_messages(
    message: BaseMessage | ToolMessage,
) -> BaseMessage | HumanToolResultMessage | AIToolCallMessage:
    match message:
        case (
            SystemMessage()
            | HumanRegularMessage()
            | AIRegularMessage()
            | HumanToolResultMessage()
            | AIToolCallMessage()
        ):
            return message
        case AIFunctionCallMessage():
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
        case HumanFunctionResultMessage():
            return HumanToolResultMessage(
                id=message.name, content=message.content
            )
        case _:
            assert_never(message)
