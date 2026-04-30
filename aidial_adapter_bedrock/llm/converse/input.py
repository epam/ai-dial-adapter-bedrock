import json
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import assert_never

from aidial_adapter_anthropic.adapter import UserError, ValidationError
from aidial_adapter_anthropic.dial.request import ToolsConfig, is_system_role
from aidial_adapter_anthropic.dial.resource import (
    AttachmentResource,
    DialResource,
    Resource,
    UnsupportedContentType,
    URLResource,
)
from aidial_sdk.chat_completion import (
    Attachment,
    InputFile,
    MessageContentAudioPart,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentTextPart,
)
from aidial_sdk.chat_completion import FunctionCall as DialFunctionCall
from aidial_sdk.chat_completion import Message as DialMessage
from aidial_sdk.chat_completion import Role as DialRole
from aidial_sdk.chat_completion import Tool as DialTool
from aidial_sdk.chat_completion import ToolCall as DialToolCall
from aidial_sdk.chat_completion import ToolChoice as DialToolChoice
from aidial_sdk.chat_completion.request import MessageContentRefusalPart
from aidial_sdk.exceptions import RuntimeServerError

from aidial_adapter_bedrock.dial_api.storage import FileStorage
from aidial_adapter_bedrock.llm.converse.constants import (
    CONVERSE_DOCUMENT_TYPE_TO_MIME,
    CONVERSE_IMAGE_TYPE_TO_MIME,
    DOCUMENT_MIME_TO_CONVERSE_TYPE,
    IMAGE_MIME_TO_CONVERSE_TYPE,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseCachePoint,
    ConverseCachePointPart,
    ConverseContentPart,
    ConverseDocumentPart,
    ConverseDocumentPartConfig,
    ConverseDocumentType,
    ConverseImagePart,
    ConverseImagePartConfig,
    ConverseImageType,
    ConverseMessage,
    ConverseRole,
    ConverseTextPart,
    ConverseToolConfig,
    ConverseToolResultPart,
    ConverseTools,
    ConverseToolSpec,
    ConverseToolUsePart,
)
from aidial_adapter_bedrock.utils.list import group_by
from aidial_adapter_bedrock.utils.list_projection import ListProjection
from aidial_adapter_bedrock.utils.text import capitalize


def to_converse_role(role: DialRole) -> ConverseRole:
    """
    Converse API accepts only 'user' and 'assistant' roles
    """
    match role:
        case DialRole.USER | DialRole.TOOL | DialRole.FUNCTION:
            return ConverseRole.USER
        case DialRole.ASSISTANT:
            return ConverseRole.ASSISTANT
        case DialRole.SYSTEM | DialRole.DEVELOPER:
            raise ValidationError(
                "System or developer messages are not allowed"
            )
        case _:
            assert_never(role)


def to_converse_tools(
    tools_config: ToolsConfig | None, ensure_non_empty_descriptions: bool
) -> ConverseTools | None:
    if tools_config is None or not tools_config.tools:
        return None

    tools: list[ConverseToolSpec | ConverseCachePointPart] = []
    for tool in tools_config.tools:
        function = tool.function
        tools_spec: ConverseToolConfig = {
            "name": function.name,
            "inputSchema": {
                "json": function.parameters
                or {"type": "object", "properties": {}}
            },
        }

        if ensure_non_empty_descriptions:
            tools_spec["description"] = function.description or " "
        elif function.description:
            tools_spec["description"] = function.description

        tools.append({"toolSpec": tools_spec})

        if cache_point_part := _get_cache_point_part(tool):
            tools.append(cache_point_part)

    match tools_config.tool_choice:
        case DialToolChoice(function=function):
            tool_choice = {"tool": {"name": function.name}}
        case "required":
            tool_choice = {"any": {}}
        case "auto":
            tool_choice = {"auto": {}}
        case "none":
            raise ValidationError(
                "tool_choice=none isn't supported by Converse API"
            )
        case _:
            assert_never(tools_config.tool_choice)

    return {"tools": tools, "toolChoice": tool_choice}


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


def sanitize_document_name(name: str) -> str:
    """
    The name must:
    - Be between 1-200 characters long
    - Only contain alphanumeric chars, spaces, hyphens, parentheses, and square brackets
    - Not have consecutive spaces
    """
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^a-zA-Z0-9\-\(\)\[\] _]", "_", name)
    return name[:200]


def to_converse_multi_modal_part(
    resource: Resource,
    name: str | None = None,
) -> ConverseImagePart | ConverseDocumentPart:
    if converse_type := IMAGE_MIME_TO_CONVERSE_TYPE.get(resource.type):
        return ConverseImagePart(
            image=ConverseImagePartConfig(
                format=converse_type,
                source={"bytes": resource.data},
            )
        )
    elif converse_type := DOCUMENT_MIME_TO_CONVERSE_TYPE.get(resource.type):
        return ConverseDocumentPart(
            document=ConverseDocumentPartConfig(
                format=converse_type,
                name=sanitize_document_name(name or str(uuid.uuid4())),
                source={"bytes": resource.data},
            )
        )
    else:
        raise UnsupportedContentType(
            message="Unknown multi-modal type",
            type=resource.type,
            supported_types=[],
        )


async def _get_converse_message_content(
    message: DialMessage,
    storage: FileStorage | None,
    supported_image_types: list[ConverseImageType],
    supported_document_types: list[ConverseDocumentType],
) -> list[ConverseContentPart]:
    image_mime_types = [
        CONVERSE_IMAGE_TYPE_TO_MIME[t] for t in supported_image_types
    ]
    document_mime_types = [
        CONVERSE_DOCUMENT_TYPE_TO_MIME[t] for t in supported_document_types
    ]

    def _unsupported_multi_modal_error(content_type: str) -> str:
        message = f"Unsupported attachment type: {content_type}\n"
        if not supported_image_types and not supported_document_types:
            return message + "Model does not support multi-modal"

        if supported_image_types:
            message += f"Supported image types: {', '.join([t.value for t in supported_image_types])}\n"
        else:
            message += "Images are not supported\n"

        if supported_document_types:
            message += f"Supported document types: {', '.join([t.value for t in supported_document_types])}"
        else:
            message += "Documents are not supported"
        return message

    if message.role == DialRole.FUNCTION:
        return [function_result_to_content_part(message)]
    elif message.role == DialRole.TOOL:
        return [tool_result_to_content_part(message)]

    content: list[ConverseContentPart] = []
    dial_resources: list[DialResource] = []

    match message.content:
        case str():
            if message.content:
                content.append({"text": message.content})
        case list():
            for part in message.content:
                match part:
                    case MessageContentTextPart():
                        if part.text:
                            content.append({"text": part.text})
                    case MessageContentImagePart():
                        dial_resources.append(
                            URLResource(
                                url=part.image_url.url,
                                supported_types=image_mime_types,
                            )
                        )
                    case MessageContentFilePart(file=file):
                        attachment = _file_content_part_to_attachment(file)
                        dial_resources.append(
                            AttachmentResource(
                                attachment=attachment,
                                entity_name="file content part",
                                supported_types=image_mime_types
                                + document_mime_types,
                            )
                        )
                    case MessageContentAudioPart():
                        raise ValidationError(
                            "Audio content parts aren't supported"
                        )
                    case MessageContentRefusalPart():
                        raise ValidationError(
                            "Refusal content parts aren't supported"
                        )
                    case _:
                        assert_never(part)

        case None:
            pass
        case _:
            assert_never(message.content)

    if message.custom_content:
        for attachment in message.custom_content.attachments or []:
            dial_resources.append(
                AttachmentResource(
                    attachment=attachment,
                    supported_types=image_mime_types + document_mime_types,
                )
            )

    for dial_resource in dial_resources:
        try:
            resource = await dial_resource.download(storage)

            name = None
            if isinstance(dial_resource, AttachmentResource):
                name = dial_resource.attachment.title

            content.append(to_converse_multi_modal_part(resource, name=name))
        except UnsupportedContentType as e:
            msg = _unsupported_multi_modal_error(e.type)
            raise UserError(error_message=msg) from None

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

    if not content:
        content.append({"text": ""})

    if cache_point_part := _get_cache_point_part(message):
        content.append(cache_point_part)

    return content


def _file_content_part_to_attachment(file: InputFile) -> Attachment:
    if (file_data := file.file_data) is None:
        raise ValidationError("File content part must have file_data field")

    resource = None
    with suppress(Exception):
        resource = Resource.from_data_url(file_data) or Resource.from_base64(
            "application/pdf", file_data
        )

    if resource is None:
        raise ValidationError(
            f"Invalid file content part: file_data must be a valid data URL or base64 string: {file_data[:30]}..."
        ) from None

    return Attachment(data=resource.data_base64, type=resource.type)


async def to_converse_message(
    message: DialMessage,
    storage: FileStorage | None = None,
    supported_image_types: list[ConverseImageType] | None = None,
    supported_document_types: list[ConverseDocumentType] | None = None,
) -> ConverseMessage:
    return {
        "role": to_converse_role(message.role),
        "content": await _get_converse_message_content(
            message,
            storage,
            supported_image_types or [],
            supported_document_types or [],
        ),
    }


@dataclass
class ExtractSystemPromptResult:
    system_messages: list[ConverseTextPart | ConverseCachePointPart]
    system_message_count: int
    non_system_messages: list[DialMessage]


def _get_cache_point_part(
    message: DialMessage | DialTool,
) -> ConverseCachePointPart | None:
    if not (cf := message.custom_fields) or not cf.cache_breakpoint:
        return None
    return ConverseCachePointPart(cachePoint=ConverseCachePoint(type="default"))


def extract_converse_system_prompt(
    messages: list[DialMessage],
) -> ExtractSystemPromptResult:
    system_messages: list[ConverseTextPart | ConverseCachePointPart] = []
    found_non_system = False
    system_messages_count = 0
    non_system_messages: list[DialMessage] = []

    for msg in messages:
        role = msg.role.value.lower().capitalize()
        if is_system_role(msg.role):
            if found_non_system:
                raise ValidationError(
                    f"{role} message can only follow system or developer message"
                )
            system_messages_count += 1

            match msg.content:
                case str():
                    system_messages.append(ConverseTextPart(text=msg.content))
                case list():
                    for part in msg.content:
                        match part:
                            case MessageContentTextPart(text=text):
                                system_messages.append(
                                    ConverseTextPart(text=text)
                                )
                            case MessageContentImagePart():
                                raise ValidationError(
                                    capitalize(
                                        f"{role} message cannot contain image content parts"
                                    )
                                )
                            case MessageContentFilePart():
                                raise ValidationError(
                                    capitalize(
                                        f"{role} message cannot contain file content parts"
                                    )
                                )
                            case MessageContentAudioPart():
                                raise ValidationError(
                                    capitalize(
                                        f"{role} message cannot contain audio content parts"
                                    )
                                )
                            case MessageContentRefusalPart():
                                raise ValidationError(
                                    capitalize(
                                        f"{role} message cannot contain refusal content parts"
                                    )
                                )
                            case _:
                                assert_never(part)
                case None:
                    pass
                case _:
                    assert_never(msg.content)

            if cache_point := _get_cache_point_part(msg):
                system_messages.append(cache_point)

        else:
            found_non_system = True
            non_system_messages.append(msg)

    return ExtractSystemPromptResult(
        system_messages=system_messages,
        system_message_count=system_messages_count,
        non_system_messages=non_system_messages,
    )


async def to_converse_messages(
    messages: list[DialMessage],
    storage: FileStorage | None = None,
    supported_image_types: list[ConverseImageType] | None = None,
    supported_document_types: list[ConverseDocumentType] | None = None,
    # Offset for system messages at the beginning
    start_offset: int = 0,
) -> ListProjection[ConverseMessage]:
    def _merge(
        a: tuple[ConverseMessage, set[int]],
        b: tuple[ConverseMessage, set[int]],
    ) -> tuple[ConverseMessage, set[int]]:
        (msg1, set1), (msg2, set2) = a, b

        content1 = msg1["content"]
        content2 = msg2["content"]

        return {
            "role": msg1["role"],
            "content": content1 + content2,
        }, set1 | set2

    converted = [
        (
            await to_converse_message(
                msg, storage, supported_image_types, supported_document_types
            ),
            {idx},
        )
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
