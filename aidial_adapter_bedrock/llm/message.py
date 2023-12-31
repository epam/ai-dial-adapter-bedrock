from typing import List, Optional, Union

from aidial_sdk.chat_completion import FunctionCall, Message, Role, ToolCall
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.exceptions import ValidationError


class SystemMessage(BaseModel):
    content: str


class HumanRegularMessage(BaseModel):
    content: str


class HumanToolResultMessage(BaseModel):
    id: str
    content: str


class HumanFunctionResultMessage(BaseModel):
    name: str
    content: str


class AIRegularMessage(BaseModel):
    content: str


class AIToolCallMessage(BaseModel):
    calls: List[ToolCall]


class AIFunctionCallMessage(BaseModel):
    call: FunctionCall


BaseMessage = Union[SystemMessage, HumanRegularMessage, AIRegularMessage]
ToolMessage = Union[
    HumanToolResultMessage,
    HumanFunctionResultMessage,
    AIToolCallMessage,
    AIFunctionCallMessage,
]


def _parse_assistant_message(
    content: Optional[str],
    function_call: Optional[FunctionCall],
    tool_calls: Optional[List[ToolCall]],
) -> BaseMessage | ToolMessage:
    if content is not None and function_call is None and tool_calls is None:
        return AIRegularMessage(content=content)

    if content is None and function_call is not None and tool_calls is None:
        return AIFunctionCallMessage(call=function_call)

    if content is None and function_call is None and tool_calls is not None:
        return AIToolCallMessage(calls=tool_calls)

    raise ValidationError(
        "Assistant message must have one and only one of the following fields not-none: "
        f"content (is none: {content is None}), "
        f"function_call (is none: {function_call is None}), "
        f"tool_calls (is none: {tool_calls is None})"
    )


def parse_message(msg: Message) -> BaseMessage | ToolMessage:
    match msg:
        case Message(role=Role.SYSTEM, content=content) if content is not None:
            return SystemMessage(content=content)
        case Message(role=Role.USER, content=content) if content is not None:
            return HumanRegularMessage(content=content)
        case Message(
            role=Role.ASSISTANT,
            content=content,
            function_call=function_call,
            tool_calls=tool_calls,
        ):
            return _parse_assistant_message(content, function_call, tool_calls)
        case Message(
            role=Role.FUNCTION, name=name, content=content
        ) if content is not None and name is not None:
            return HumanFunctionResultMessage(name=name, content=content)
        case Message(
            role=Role.TOOL, tool_call_id=id, content=content
        ) if content is not None and id is not None:
            return HumanToolResultMessage(id=id, content=content)
        case _:
            raise ValidationError("Unknown message type or invalid message")
