import json
from typing import Any, Dict, List, assert_never

from aidial_sdk.chat_completion import FinishReason as DialFinishReason
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
    ValidationError,
)
from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.llm.converse.constants import (
    CONVERSE_TO_DIAL_FINISH_REASON,
    DIAL_TO_CONVERSE_FINISH_REASON,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseContentPart,
    ConverseMessage,
    ConverseRole,
    ConverseStopReason,
    ConverseTools,
)


def to_dial_finish_reason(
    converse_stop_reason: ConverseStopReason,
) -> DialFinishReason:
    if converse_stop_reason not in CONVERSE_TO_DIAL_FINISH_REASON.keys():
        raise RuntimeServerError(
            f"Unsupported converse stop reason: {converse_stop_reason}"
        )
    return CONVERSE_TO_DIAL_FINISH_REASON[converse_stop_reason]


def to_converse_finish_reason(
    dial_finish_reason: DialFinishReason,
) -> ConverseStopReason:
    if dial_finish_reason not in DIAL_TO_CONVERSE_FINISH_REASON.keys():
        raise RuntimeServerError(
            f"Unsupported DIAL stop reason: {dial_finish_reason.value}"
        )
    return DIAL_TO_CONVERSE_FINISH_REASON[dial_finish_reason]


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
                "inputSchema": (
                    {
                        "json": (
                            {
                                "type": "object",
                                "properties": function.parameters or {},
                            }
                        )
                    }
                ),
            }
        }
        tools.append(tool)

    return {
        "tools": tools,
    }


def function_call_to_content_part(
    dial_call: DialFunctionCall,
) -> ConverseContentPart:
    return {
        "toolUse": {
            "toolUseId": dial_call.name,
            "name": dial_call.name,
            "input": json.loads(dial_call.arguments),
        }
    }


def tool_call_to_content_part(
    dial_call: DialToolCall,
) -> ConverseContentPart:
    return {
        "toolUse": {
            "toolUseId": dial_call.id,
            "name": dial_call.function.name,
            "input": json.loads(dial_call.function.arguments),
        }
    }


def function_result_to_content_part(
    message: DialMessage,
) -> ConverseContentPart:
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


def tool_result_to_content_part(
    message: DialMessage,
) -> ConverseContentPart:
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


async def _get_converse_message_content(
    message: DialMessage,
    storage: FileStorage | None,
    supported_image_types: list[str] | None = None,
) -> List[ConverseContentPart]:
    if message.function_call:
        return [function_call_to_content_part(message.function_call)]
    elif message.tool_calls:
        return [
            tool_call_to_content_part(tool_call)
            for tool_call in message.tool_calls
        ]
    elif message.role == DialRole.FUNCTION:
        return [function_result_to_content_part(message)]
    elif message.role == DialRole.TOOL:
        return [tool_result_to_content_part(message)]

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

    return content


async def to_converse_message(
    message: DialMessage,
    storage: FileStorage | None,
    supported_image_types: list[str] | None = None,
) -> ConverseMessage:

    return {
        "role": to_converse_role(message.role),
        "content": await _get_converse_message_content(
            message, storage, supported_image_types
        ),
    }


def get_converse_system_prompt(
    messages: List[DialMessage],
) -> Dict[str, Any] | None:
    if any(msg.role == DialRole.SYSTEM for msg in messages[1:]):
        raise ValidationError(
            "System message is only allowed as the first message"
        )

    if messages[0].role == DialRole.SYSTEM:
        return {"text": messages[0].content}
    return None
