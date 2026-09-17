import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Self, cast

from aidial_sdk.chat_completion.request import (
    CacheBreakpoint,
    ChatCompletionRequestCustomFields,
    FunctionCall,
    FunctionChoice,
    ImageURL,
    InputFile,
    MessageContentFilePart,
    MessageContentImagePart,
    MessageContentPart,
    MessageContentTextPart,
    MessageCustomFields,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaObject,
    Role,
    StaticTool,
    StreamOptions,
    ToolCall,
    ToolCustomFields,
)
from aidial_sdk.chat_completion.request import Function as SdkFunction
from aidial_sdk.chat_completion.request import Message as SdkMessage
from aidial_sdk.chat_completion.request import Tool as SdkTool
from aidial_sdk.chat_completion.request import ToolChoice as SdkToolChoice
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam as CacheControl,
)
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaJSONOutputFormatParam,
    BetaMessageParam,
    BetaRequestDocumentBlockParam,
    BetaToolChoiceParam,
    BetaToolParam,
    BetaToolResultBlockParam,
    BetaToolUnionParam,
)
from anthropic.types.beta.message_create_params import MessageCreateParams
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel

from aidial_adapter_bedrock.anthropic_translator.chat_completions.cache_breakpoints import (
    cache_breakpoint,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    resolve_effort,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_JSON_SCHEMA_NAME: str = "response"

_SERVICE_TIERS: dict[str, str] = {"auto": "auto", "standard_only": "default"}


class CoreChatCompletionRequest(BaseModel):
    model: str
    messages: list[SdkMessage]
    max_completion_tokens: int
    custom_fields: ChatCompletionRequestCustomFields | None
    tools: list[SdkTool | StaticTool] | None
    tool_choice: Literal["auto", "none", "required"] | SdkToolChoice | None
    parallel_tool_calls: bool | None
    reasoning_effort: ReasoningEffort | Literal["max"]
    response_format: ResponseFormat | None
    stop: list[str] | None
    temperature: float | None
    top_p: float | None
    user: str | None
    service_tier: str | None
    stream: bool = False
    stream_options: StreamOptions | None = None


@dataclass
class SystemPrompt:
    texts: list[str]
    cache_controls: list[CacheControl]

    @property
    def text(self) -> str:
        return "\n\n".join(self.texts)

    def extend(self, other: Self) -> None:
        self.texts.extend(other.texts)
        self.cache_controls.extend(other.cache_controls)


def to_chat_completions_request(
    req: MessageCreateParams,
    deployment: str,
    aliases: ToolNameAliases,
) -> CoreChatCompletionRequest:
    messages: list[SdkMessage] = _convert_messages(req, aliases)

    _warn_dropped(req)

    return CoreChatCompletionRequest(
        model=deployment,
        messages=messages,
        custom_fields=(
            ChatCompletionRequestCustomFields(
                configuration={"enable_citations": True}
            )
            if _any_citations_enabled(req["messages"])
            else None
        ),
        tools=_convert_tools(req.get("tools"), aliases) or None,
        tool_choice=_convert_tool_choice(req.get("tool_choice"), aliases),
        parallel_tool_calls=_convert_parallel_tool_calls(
            req.get("tool_choice")
        ),
        reasoning_effort=resolve_effort(req),
        response_format=_convert_response_format(req),
        stop=list(req.get("stop_sequences") or []) or None,
        max_completion_tokens=req["max_tokens"],
        temperature=req.get("temperature"),
        top_p=req.get("top_p"),
        user=(req.get("metadata", {}).get("user_id") or None)
        if req.get("metadata")
        else None,
        service_tier=_convert_service_tier(req),
    )


def _convert_messages(
    req: MessageCreateParams,
    aliases: ToolNameAliases,
) -> list[SdkMessage]:
    messages: list[SdkMessage] = []

    system: SystemPrompt = _collect_system(req)
    if system.texts:
        system_message: SdkMessage = SdkMessage(
            role=Role.SYSTEM, content=system.text
        )
        _mark_message(system_message, system.cache_controls)
        messages.append(system_message)

    converted: list[SdkMessage]
    for message in req["messages"]:
        match message["role"]:
            case "user":
                converted = _convert_user_message(message["content"])
            case "assistant":
                converted = _convert_assistant_message(
                    message["content"], aliases
                )
            case _:
                continue

        controls: list[CacheControl] = _cache_controls(message["content"])
        for converted_message in converted:
            _mark_message(converted_message, controls)
        messages.extend(converted)

    if (control := req.get("cache_control")) is not None:
        for message in reversed(messages):
            if message.role == Role.USER:
                _mark_message(message, [control])
                break
    return messages


def _warn_dropped(req: MessageCreateParams) -> None:
    if req.get("top_k") is not None:
        log.debug("Dropping unsupported 'top_k' parameter")
    if req.get("mcp_servers"):
        log.warning("Dropping 'mcp_servers': no Chat Completions equivalent")
    if req.get("container"):
        log.warning("Dropping 'container': no Chat Completions equivalent")
    if req.get("inference_geo"):
        log.warning("Dropping 'inference_geo': no Chat Completions equivalent")
    if req.get("context_management"):
        log.warning(
            "Dropping 'context_management': no Chat Completions equivalent"
        )


def _convert_service_tier(req: MessageCreateParams) -> str | None:
    if (tier := req.get("service_tier")) is None:
        return None
    return _SERVICE_TIERS[tier]


def _convert_response_format(req: MessageCreateParams) -> ResponseFormat | None:
    output_format: BetaJSONOutputFormatParam | None = (
        req.get("output_config") or {}
    ).get("format")
    if output_format is None:
        return None
    if output_format.get("type") != "json_schema":
        log.warning(
            "Dropping unsupported output_config.format type: %s",
            output_format.get("type"),
        )
        return None
    schema: dict[str, object] | None = output_format.get("schema")
    if not schema:
        log.warning("Dropping output_config.format: missing 'schema'")
        return None
    return ResponseFormatJsonSchema(
        type="json_schema",
        json_schema=ResponseFormatJsonSchemaObject(
            name=_JSON_SCHEMA_NAME,
            schema=schema,
            strict=True,
        ),
    )


def _cache_controls(content: object) -> list[CacheControl]:
    return [
        control
        for block in _blocks(content)
        if isinstance(block, dict) and (control := block.get("cache_control"))
    ]


def _mark_message(
    message: SdkMessage,
    controls: list[CacheControl],
) -> None:
    marker: CacheBreakpoint | None = cache_breakpoint(controls)
    if marker is None:
        return
    if message.custom_fields is None:
        message.custom_fields = MessageCustomFields(cache_breakpoint=marker)
        return
    current: CacheBreakpoint | None = message.custom_fields.cache_breakpoint
    if current and (current.expire_at or "") > (marker.expire_at or ""):
        return
    message.custom_fields.cache_breakpoint = marker


def _mark_tool(
    tool: SdkTool,
    controls: list[CacheControl],
) -> None:
    if (marker := cache_breakpoint(controls)) is not None:
        tool.custom_fields = ToolCustomFields(cache_breakpoint=marker)


def _any_citations_enabled(messages: Iterable[BetaMessageParam]) -> bool:
    for message in messages:
        for block in _blocks(message["content"]):
            if not isinstance(block, dict):
                continue
            if (
                block["type"] == "document"
                and (citations := block.get("citations"))
                and citations.get("enabled")
            ):
                return True
    return False


def _blocks(content: object) -> list[BetaContentBlockParam]:
    if not isinstance(content, list):
        return []
    return cast(list[BetaContentBlockParam], content)


def _system_text(content: object) -> SystemPrompt:
    if isinstance(content, str):
        return SystemPrompt(
            texts=[content] if content else [], cache_controls=[]
        )

    texts: list[str] = []
    for block in _blocks(content):
        if not isinstance(block, dict):
            continue
        match block["type"]:
            case "text":
                if text := block.get("text"):
                    texts.append(text)
            case "mid_conv_system":
                continue
            case unsupported:
                log.warning(
                    "Dropping unsupported system content block: %s", unsupported
                )
    return SystemPrompt(
        texts=texts, cache_controls=_cache_controls(content) if texts else []
    )


def _collect_system(req: MessageCreateParams) -> SystemPrompt:
    system: SystemPrompt = _system_text(req.get("system"))

    for message in req["messages"]:
        if message.get("role") == "system":
            system.extend(_system_text(message["content"]))

        for block in _blocks(message["content"]):
            if not isinstance(block, dict):
                continue
            if block["type"] != "mid_conv_system":
                continue
            if control := block.get("cache_control"):
                system.cache_controls.append(control)
            system.extend(_system_text(block.get("content")))

    return system


def _convert_user_message(
    content: str | Iterable[BetaContentBlockParam],
) -> list[SdkMessage]:
    if isinstance(content, str):
        if not content:
            return []
        return [SdkMessage(role=Role.USER, content=content)]

    tool_messages: list[SdkMessage] = []
    parts: list[MessageContentPart] = []

    for block in _blocks(content):
        if not isinstance(block, dict):
            continue
        match block["type"]:
            case "tool_result":
                tool_messages.append(_tool_result_message(block))
                parts.extend(_tool_result_images(block.get("content")))
            case "text":
                if block.get("text"):
                    parts.append(
                        MessageContentTextPart(
                            type="text", text=block.get("text")
                        )
                    )
            case "image":
                if part := _image_part(block):
                    parts.append(part)
            case "document":
                if part := _document_part(block):
                    parts.append(part)
            case "mid_conv_system":
                continue
            case unsupported:
                log.warning(
                    "Dropping unsupported user content block: %s", unsupported
                )

    if parts:
        tool_messages.append(SdkMessage(role=Role.USER, content=parts))
    return tool_messages


def _tool_result_message(block: BetaToolResultBlockParam) -> SdkMessage:
    text: str = _tool_result_text(block.get("content"))
    return SdkMessage(
        role=Role.TOOL,
        tool_call_id=block.get("tool_use_id"),
        content=f"Error: {text}" if block.get("is_error") else text,
    )


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        text
        for sub in _blocks(content)
        if isinstance(sub, dict)
        and sub["type"] == "text"
        and (text := sub.get("text"))
    )


def _tool_result_images(content: object) -> list[MessageContentPart]:
    return [
        part
        for sub in _blocks(content)
        if isinstance(sub, dict)
        and sub["type"] == "image"
        and (part := _image_part(sub))
    ]


def _convert_assistant_message(
    content: str | Iterable[BetaContentBlockParam],
    aliases: ToolNameAliases,
) -> list[SdkMessage]:
    if isinstance(content, str):
        return (
            [SdkMessage(role=Role.ASSISTANT, content=content)]
            if content
            else []
        )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        match block["type"]:
            case "text":
                if block.get("text"):
                    text_parts.append(block.get("text"))
            case "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id") or "",
                        type="function",
                        function=FunctionCall(
                            name=aliases.to_upstream(block.get("name") or ""),
                            arguments=json.dumps(block.get("input") or {}),
                        ),
                    )
                )
            case "thinking" | "redacted_thinking" | "mid_conv_system":
                continue
            case unsupported:
                log.warning(
                    "Dropping unsupported assistant content block: %s",
                    unsupported,
                )

    if not text_parts and not tool_calls:
        return []

    return [
        SdkMessage(
            role=Role.ASSISTANT,
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
        )
    ]


def _image_part(block: BetaImageBlockParam) -> MessageContentPart | None:
    source = block["source"]
    stype: str | None = source.get("type")
    if source["type"] == "base64":
        media_type: str = source.get("media_type") or "image/png"
        data = source["data"]
        return MessageContentImagePart(
            type="image_url",
            image_url=ImageURL(url=f"data:{media_type};base64,{data}"),
        )
    if source["type"] == "url" and (url := source["url"]):
        return MessageContentImagePart(
            type="image_url", image_url=ImageURL(url=url)
        )
    log.warning("Dropping image block with source type: %s", stype)
    return None


def _document_part(
    block: BetaRequestDocumentBlockParam,
) -> MessageContentPart | None:
    source = block["source"]
    stype: str | None = source.get("type")
    filename: str = block.get("title") or "document.pdf"
    if source["type"] == "base64":
        media_type: str = source.get("media_type") or "application/pdf"
        data = source["data"]
        return MessageContentFilePart(
            type="file",
            file=InputFile(
                filename=filename,
                file_data=f"data:{media_type};base64,{data}",
            ),
        )
    if source["type"] == "text" and (text_data := source["data"]):
        return MessageContentTextPart(type="text", text=text_data)

    log.warning("Dropping document block with source type: %s", stype)
    return None


def _convert_tools(
    tools: Iterable[BetaToolUnionParam] | None,
    aliases: ToolNameAliases,
) -> list[SdkTool | StaticTool]:
    if not tools:
        return []
    result: list[SdkTool | StaticTool] = []
    for tool in tools:
        ttype: str | None = tool.get("type")
        if ttype and ttype != "custom":
            log.warning("Dropping unsupported tool type: %s", ttype)
        elif not tool.get("name"):
            log.warning("Dropping custom tool without a name")
        else:
            tool = cast(BetaToolParam, tool)
            sdk_tool: SdkTool = SdkTool(
                type="function",
                function=SdkFunction(
                    name=aliases.to_upstream(tool.get("name")),
                    description=tool.get("description") or None,
                    parameters=_parameters(tool),
                    strict=False,
                ),
            )
            _mark_tool(
                sdk_tool,
                [control] if (control := tool.get("cache_control")) else [],
            )
            result.append(sdk_tool)
    return result


def _parameters(tool: BetaToolParam) -> dict[str, object]:
    if tool.get("input_schema") is None:
        return {"type": "object", "properties": {}}

    return {
        key: value
        for key, value in tool.get("input_schema").items()
        if key != "$schema"
    }


def _convert_tool_choice(
    tool_choice: BetaToolChoiceParam | None, aliases: ToolNameAliases
) -> Literal["auto", "none", "required"] | SdkToolChoice | None:
    if tool_choice is None:
        return None
    match tool_choice["type"]:
        case "auto":
            return "auto"
        case "any":
            return "required"
        case "none":
            return "none"
        case "tool":
            return SdkToolChoice(
                type="function",
                function=FunctionChoice(
                    name=aliases.to_upstream(tool_choice["name"])
                    if tool_choice
                    else ""
                ),
            )
        case _:
            return None


def _convert_parallel_tool_calls(
    tool_choice: BetaToolChoiceParam | None,
) -> bool | None:
    if not tool_choice or "disable_parallel_tool_use" not in tool_choice:
        return None
    return not bool(tool_choice.get("disable_parallel_tool_use"))
