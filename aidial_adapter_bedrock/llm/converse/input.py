import json
import uuid
from dataclasses import dataclass
from typing import List, Set, Tuple, assert_never

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
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseContentPart,
    ConverseDocumentPart,
    ConverseDocumentPartConfig,
    ConverseImagePart,
    ConverseImagePartConfig,
    ConverseMessage,
    ConverseRole,
    ConverseTextPart,
    ConverseToolResultPart,
    ConverseTools,
    ConverseToolSpec,
    ConverseToolUsePart,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.list import group_by
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from aidial_adapter_bedrock.utils.resource import Resource

IMAGE_MIME_TO_CONVERSE_TYPE = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

DOCUMENT_MIME_TO_CONVERSE_TYPE = {
    "application/pdf": "pdf",
    "application/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/plain": "txt",
    "text/markdown": "md",
}


def to_converse_role(role: DialRole) -> ConverseRole:
    """
    Converse API accepts only 'user' and 'assistant' roles
    """
    match role:
        case DialRole.USER | DialRole.TOOL | DialRole.FUNCTION:
            return ConverseRole.USER
        case DialRole.ASSISTANT:
            return ConverseRole.ASSISTANT
        case DialRole.SYSTEM:
            raise ValidationError("System messages are not allowed")
        case _:
            assert_never(role)


def to_converse_tools(tools_config: ToolsConfig) -> ConverseTools:
    tools: list[ConverseToolSpec] = []
    for function in tools_config.functions:
        tools.append(
            {
                "toolSpec": {
                    "name": function.name,
                    "description": function.description or "",
                    "inputSchema": {
                        "json": function.parameters
                        or {"type": "object", "properties": {}}
                    },
                }
            }
        )

    return {
        "tools": tools,
        "toolChoice": ({"any": {}} if tools_config.required else {"auto": {}}),
    }


def function_call_to_content_part(
    dial_call: DialFunctionCall,
) -> ConverseToolUsePart:
    return {
        "toolUse": {
            "toolUseId": dial_call.name,
            "name": dial_call.name,
            "input": json.loads(dial_call.arguments),
        }
    }


def tool_call_to_content_part(
    dial_call: DialToolCall,
) -> ConverseToolUsePart:
    return {
        "toolUse": {
            "toolUseId": dial_call.id,
            "name": dial_call.function.name,
            "input": json.loads(dial_call.function.arguments),
        }
    }


def function_result_to_content_part(
    message: DialMessage,
) -> ConverseToolResultPart:
    if message.role != DialRole.FUNCTION:
        raise RuntimeServerError(
            "Function result message is expected to have function role"
        )
    if not message.name or not isinstance(message.content, str):
        raise RuntimeServerError(
            "Function result message is expected to have function name and plain text content"
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
) -> ConverseToolResultPart:
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


def to_converse_multi_modal_part(
    resource: Resource,
) -> ConverseImagePart | ConverseDocumentPart:
    if resource.type in IMAGE_MIME_TO_CONVERSE_TYPE:
        return ConverseImagePart(
            image=ConverseImagePartConfig(
                format=IMAGE_MIME_TO_CONVERSE_TYPE[resource.type],
                source={"bytes": resource.data},
            )
        )
    elif resource.type in DOCUMENT_MIME_TO_CONVERSE_TYPE:
        return ConverseDocumentPart(
            document=ConverseDocumentPartConfig(
                format=DOCUMENT_MIME_TO_CONVERSE_TYPE[resource.type],
                # Passing generic name, since provider explicitly stated,
                # that this field is vulnerable for prompt injection, because the model might
                # inadvertently interpret it as instructions.
                name=f"{uuid.uuid4()}.{DOCUMENT_MIME_TO_CONVERSE_TYPE[resource.type]}",
                source={"bytes": resource.data},
            )
        )
    else:
        raise RuntimeServerError(
            f"Unsupported multi-modal type: {resource.type}"
        )


async def _get_converse_message_content(
    message: DialMessage,
    storage: FileStorage | None,
    supported_image_types: list[str] | None = None,
) -> List[ConverseContentPart]:

    if message.role == DialRole.FUNCTION:
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
                        content.append(to_converse_multi_modal_part(resource))

        case None:
            pass
        case _:
            assert_never(message.content)

    if message.custom_content and message.custom_content.attachments:
        for attachment in message.custom_content.attachments:
            resource = await AttachmentResource(attachment=attachment).download(
                storage
            )
            content.append(to_converse_multi_modal_part(resource))
    if message.function_call and message.tool_calls:
        raise ValidationError(
            "You cannot use both function call and tool calls in the same message"
        )
    elif message.function_call:
        content.append(function_call_to_content_part(message.function_call))
    elif message.tool_calls:
        content.extend(
            [
                tool_call_to_content_part(tool_call)
                for tool_call in message.tool_calls
            ]
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


@dataclass
class ExtractSystemPromptResult:
    system_prompt: ConverseTextPart | None
    system_message_count: int
    non_system_messages: List[DialMessage]


def extract_converse_system_prompt(
    messages: List[DialMessage],
) -> ExtractSystemPromptResult:
    system_msgs = []
    found_non_system = False
    system_messages_count = 0
    non_system_messages = []

    for msg in messages:
        if msg.role == DialRole.SYSTEM:
            if found_non_system:
                raise ValidationError(
                    "A system message can only follow another system message"
                )
            system_messages_count += 1
            match msg.content:
                case str():
                    system_msgs.append(msg.content)
                case list():
                    for part in msg.content:
                        match part:
                            case MessageContentTextPart():
                                system_msgs.append(part.text)
                            case MessageContentImagePart():
                                raise ValidationError(
                                    "System messages cannot contain images"
                                )
                case None:
                    pass
                case _:
                    assert_never(msg.content)
        else:
            found_non_system = True
            non_system_messages.append(msg)
    combined = "\n\n".join(msg for msg in system_msgs if msg)
    return ExtractSystemPromptResult(
        system_prompt=ConverseTextPart(text=combined) if combined else None,
        system_message_count=system_messages_count,
        non_system_messages=non_system_messages,
    )


async def to_converse_messages(
    messages: List[DialMessage],
    storage: FileStorage | None,
    # Offset for system messages at the beginning
    start_offset: int = 0,
) -> ListProjection[ConverseMessage]:
    def _merge(
        a: Tuple[ConverseMessage, Set[int]],
        b: Tuple[ConverseMessage, Set[int]],
    ) -> Tuple[ConverseMessage, Set[int]]:
        (msg1, set1), (msg2, set2) = a, b

        content1 = msg1["content"]
        content2 = msg2["content"]

        return {
            "role": msg1["role"],
            "content": content1 + content2,
        }, set1 | set2

    converted = [
        (await to_converse_message(msg, storage), set([idx]))
        for idx, msg in enumerate(messages, start=start_offset)
    ]

    # Merge messages with the same roles to achieve an alternation of user-assistant roles.
    return ListProjection(
        group_by(
            lst=converted,
            key=lambda msg: msg[0]["role"],
            init=lambda msg: msg,
            merge=_merge,
        )
    )
