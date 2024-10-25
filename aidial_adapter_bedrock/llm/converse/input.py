import json
from enum import Enum
from typing import Any, Dict, TypedDict, assert_never

from aidial_sdk.chat_completion import FunctionCall as DialFunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import (
    MessageContentImagePart,
    MessageContentTextPart,
)
from aidial_sdk.chat_completion import Role as DialRole
from aidial_sdk.chat_completion import ToolCall as DialToolCall
from aidial_sdk.exceptions import RuntimeServerError

from aidial_adapter_bedrock.dial_api.request import ToolsConfig
from aidial_adapter_bedrock.dial_api.resource import (
    AttachmentResource,
    URLResource,
)
from aidial_adapter_bedrock.dial_api.storage import FileStorage


class ConverseRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConverseToolConfig(TypedDict):
    name: str
    description: str
    inputSchema: dict


class ConverseTools(TypedDict):
    tools: list[ConverseToolConfig]


class ConverseToolUseConfig(TypedDict):
    toolUseId: str
    name: str
    input: str


class ConverseToolUse(TypedDict):
    toolUse: ConverseToolUseConfig


class ConverseToolResultConfig(TypedDict):
    toolUseId: str
    content: list[dict]
    status: str


class ConverseToolResult(TypedDict):
    toolResult: ConverseToolResultConfig


def to_converse_role(role: DialRole) -> ConverseRole:
    """
    Converse API accepts only 'user' and 'assistant' roles
    """
    match role:
        case (
            DialRole.USER | DialRole.TOOL | DialRole.FUNCTION | DialRole.SYSTEM
        ):
            return ConverseRole.USER
        case DialRole.ASSISTANT:
            return ConverseRole.ASSISTANT
        case _:
            assert_never(role)


def to_converse_tools(tools_config: ToolsConfig) -> ConverseTools:
    tools = []
    for function in tools_config.functions:
        tool = {
            "toolSpec": {
                "name": function.name,
                "description": function.description or "",
                "inputSchema": {"json": function.parameters or {}},
            }
        }
        tools.append(tool)

    return {
        "tools": tools,
    }


def dial_function_to_converse_tool_part(
    dial_call: DialFunctionCall,
) -> ConverseToolUse:
    return {
        "toolUse": {
            "toolUseId": dial_call.name,
            "name": dial_call.name,
            "input": dial_call.arguments,
        }
    }


def dial_tool_call_to_converse_tool(
    dial_call: DialToolCall,
) -> ConverseToolUse:
    return {
        "toolUse": {
            "toolUseId": dial_call.id,
            "name": dial_call.function.name,
            "input": dial_call.function.arguments,
        }
    }


def dial_function_result_to_converse_tool_result(
    message: DialMessage,
) -> ConverseToolResult:
    if message.role != DialRole.FUNCTION:
        raise RuntimeServerError(
            "Function result message is expected to have function role"
        )
    if not message.name or not message.content:
        raise RuntimeServerError(
            "Function result message is expected to have function name and content"
        )

    return {
        "toolResult": {
            "toolUseId": message.name,
            "content": [{"text": message.content}],
            "status": "success",
        }
    }


def dial_tool_result_to_converse_tool_result(
    message: DialMessage,
) -> ConverseToolResult:
    if message.role != DialRole.TOOL:
        raise RuntimeServerError(
            "Tool result message is expected to have tool role"
        )
    if not message.tool_call_id or not isinstance(message.content, str):
        raise RuntimeServerError(
            "Tool result message is expected to have tool call id and plain text content"
        )

    try:
        json_content = json.loads(message.content)
        return {
            "toolResult": {
                "toolUseId": message.tool_call_id,
                "content": [{"json": json_content}],
                "status": "success",
            }
        }
    except json.JSONDecodeError:
        return {
            "toolResult": {
                "toolUseId": message.tool_call_id,
                "content": [{"text": message.content}],
                "status": "success",
            }
        }


def to_converse_image_type(type: str) -> str:
    if type == "image/png":
        return "png"
    if type == "image/jpeg":
        return "jpeg"
    raise RuntimeServerError(f"Unsupported image type: {type}")


async def to_converse_message(
    message: DialMessage,
    storage: FileStorage | None,
    supported_image_types: list[str] | None = None,
) -> Dict[str, Any]:
    content = []
    match message.content:
        case str():
            content.append({"text": message.content})
        case list():
            for part in message.content:
                match part:
                    case MessageContentTextPart():
                        content.append({"text": part.text})
                    case MessageContentImagePart():
                        resource = await URLResource(
                            url=part.image_url.url,
                            supported_types=supported_image_types,
                        ).download(storage)
                        content.append(
                            {
                                "image": {
                                    "format": to_converse_image_type(
                                        resource.type
                                    ),
                                    "source": {
                                        "bytes": resource.data,
                                    },
                                }
                            }
                        )
        case None:
            pass
        case _:
            assert_never(message.content)

    if message.custom_content and message.custom_content.attachments:
        for attachment in message.custom_content.attachments:
            resource = await AttachmentResource(attachment=attachment).download(
                storage
            )
            content.append(
                {
                    "image": {
                        "format": to_converse_image_type(resource.type),
                        "source": {"bytes": resource.data},
                    }
                }
            )

    if message.function_call:
        content.append(
            dial_function_to_converse_tool_part(message.function_call)
        )
    elif message.tool_calls:
        content.append(dial_tool_call_to_converse_tool(message.tool_calls[0]))

    if message.role == DialRole.FUNCTION:
        content.append(dial_function_result_to_converse_tool_result(message))
    elif message.role == DialRole.TOOL:
        content.append(dial_tool_result_to_converse_tool_result(message))

    bedrock_message = {
        "role": to_converse_role(message.role).value,
        "content": content,
    }

    return bedrock_message


def get_converse_system_prompt(
    message: DialMessage,
) -> Dict[str, Any] | None:
    if message.role != DialRole.SYSTEM:
        return None

    if not isinstance(message.content, str):
        raise RuntimeServerError(
            f"System message content expected to be a plain string, got {type(message.content)}"
        )
    return {"text": message.content}
