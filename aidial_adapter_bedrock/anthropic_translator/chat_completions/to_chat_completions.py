import json
from typing import Literal, Self

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
from pydantic import BaseModel, ValidationError

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    ContentBlock,
    ContentSource,
    JsonObject,
    Message,
    MessagesRequest,
    OutputFormat,
    Tool,
    ToolChoice,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.cache_breakpoints import (
    CacheControl,
    cache_breakpoint,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    OpenAIEffort,
    resolve_effort,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.stop_emulation import (
    strips_stop_parameter,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicErrorType,
    AnthropicHTTPError,
    format_validation_error,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

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
    reasoning_effort: OpenAIEffort | None
    response_format: ResponseFormat | None
    stop: list[str] | None
    temperature: float | None
    top_p: float | None
    user: str | None
    service_tier: str | None
    stream: bool = False
    stream_options: StreamOptions | None = None


class SystemPrompt(BaseModel):
    texts: list[str]
    cache_controls: list[CacheControl]

    @property
    def text(self) -> str:
        return "\n\n".join(self.texts)

    def extend(self, other: Self) -> None:
        self.texts.extend(other.texts)
        self.cache_controls.extend(other.cache_controls)


def to_chat_completions_request(
    req: MessagesRequest,
    deployment: str,
    aliases: ToolNameAliases,
) -> CoreChatCompletionRequest:
    tlog: TranslationLog = TranslationLog("Anthropic→Chat Completions request")
    try:
        if req.max_tokens is None:
            raise AnthropicHTTPError(
                AnthropicErrorType.INVALID_REQUEST, "'max_tokens' is required"
            )

        messages: list[SdkMessage] = _convert_messages(req, aliases, tlog)

        _warn_dropped(req, tlog)

        return CoreChatCompletionRequest(
            model=deployment,
            messages=messages,
            custom_fields=(
                ChatCompletionRequestCustomFields(
                    configuration={"enable_citations": True}
                )
                if _any_citations_enabled(req.messages)
                else None
            ),
            tools=_convert_tools(req.tools, aliases, tlog) or None,
            tool_choice=_convert_tool_choice(req.tool_choice, aliases),
            parallel_tool_calls=_convert_parallel_tool_calls(req.tool_choice),
            reasoning_effort=resolve_effort(req),
            response_format=_convert_response_format(req, tlog),
            stop=None
            if strips_stop_parameter(deployment)
            else req.stop_sequences or None,
            max_completion_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            user=(req.metadata.user_id or None) if req.metadata else None,
            service_tier=_convert_service_tier(req, tlog),
        )
    except ValidationError as error:
        raise AnthropicHTTPError(
            AnthropicErrorType.INVALID_REQUEST, format_validation_error(error)
        ) from error
    finally:
        tlog.flush()


def _convert_messages(
    req: MessagesRequest,
    aliases: ToolNameAliases,
    tlog: TranslationLog,
) -> list[SdkMessage]:
    messages: list[SdkMessage] = []

    system: SystemPrompt = _collect_system(req, tlog)
    if system.texts:
        system_message: SdkMessage = SdkMessage(
            role=Role.SYSTEM, content=system.text
        )
        _mark_message(system_message, system.cache_controls, tlog)
        messages.append(system_message)

    converted: list[SdkMessage]
    for message in req.messages:
        match message.role:
            case "user":
                converted = _convert_user_message(message.content, tlog)
            case "assistant":
                converted = _convert_assistant_message(
                    message.content, aliases, tlog
                )
            case "system":
                continue
            case unknown:
                raise AnthropicHTTPError(
                    AnthropicErrorType.INVALID_REQUEST,
                    f"Unknown message role: {unknown!r}",
                )

        controls: list[CacheControl] = _cache_controls(message.content)
        for converted_message in converted:
            _mark_message(converted_message, controls, tlog)
        messages.extend(converted)

    if req.cache_control is not None:
        for message in reversed(messages):
            if message.role == Role.USER:
                _mark_message(message, [req.cache_control], tlog)
                break
    return messages


def _warn_dropped(req: MessagesRequest, tlog: TranslationLog) -> None:
    if req.top_k is not None:
        tlog.debug("Dropping unsupported 'top_k' parameter")
    if req.mcp_servers:
        tlog.warning("Dropping 'mcp_servers': no Chat Completions equivalent")
    if req.container:
        tlog.warning("Dropping 'container': no Chat Completions equivalent")
    if req.inference_geo:
        tlog.warning("Dropping 'inference_geo': no Chat Completions equivalent")
    if req.context_management:
        tlog.warning(
            "Dropping 'context_management': no Chat Completions equivalent"
        )


def _convert_service_tier(
    req: MessagesRequest, tlog: TranslationLog
) -> str | None:
    if req.service_tier is None:
        return None
    if (tier := _SERVICE_TIERS.get(req.service_tier)) is None:
        tlog.warning("Dropping unknown service_tier: %s", req.service_tier)
    return tier


def _convert_response_format(
    req: MessagesRequest, tlog: TranslationLog
) -> ResponseFormat | None:
    output_format: OutputFormat | None = (
        req.output_config.format if req.output_config else None
    )
    if output_format is None:
        return None
    if output_format.type != "json_schema":
        tlog.warning(
            "Dropping unsupported output_config.format type: %s",
            output_format.type,
        )
        return None
    schema: JsonObject | None = output_format.schema_
    if not schema:
        tlog.warning("Dropping output_config.format: missing 'schema'")
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
        if (control := block.cache_control)
    ]


def _mark_message(
    message: SdkMessage,
    controls: list[CacheControl],
    tlog: TranslationLog,
) -> None:
    marker: CacheBreakpoint | None = cache_breakpoint(controls, tlog)
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
    tlog: TranslationLog,
) -> None:
    if (marker := cache_breakpoint(controls, tlog)) is not None:
        tool.custom_fields = ToolCustomFields(cache_breakpoint=marker)


def _any_citations_enabled(messages: list[Message]) -> bool:
    for message in messages:
        for block in _blocks(message.content):
            if (
                block.type == "document"
                and block.citations
                and block.citations.enabled
            ):
                return True
    return False


def _blocks(content: object) -> list[ContentBlock]:
    if not isinstance(content, list):
        return []
    return [
        block
        if isinstance(block, ContentBlock)
        else ContentBlock.model_validate(block)
        for block in content
        if isinstance(block, ContentBlock | dict)
    ]


def _system_text(content: object, tlog: TranslationLog) -> SystemPrompt:
    if isinstance(content, str):
        return SystemPrompt(
            texts=[content] if content else [], cache_controls=[]
        )

    texts: list[str] = []
    for block in _blocks(content):
        match block.type:
            case "text":
                if text := block.text:
                    texts.append(text)
            case "mid_conv_system":
                continue
            case unsupported:
                tlog.warning(
                    "Dropping unsupported system content block: %s", unsupported
                )
    return SystemPrompt(
        texts=texts, cache_controls=_cache_controls(content) if texts else []
    )


def _collect_system(req: MessagesRequest, tlog: TranslationLog) -> SystemPrompt:
    system: SystemPrompt = _system_text(req.system, tlog)

    for message in req.messages:
        if message.role == "system":
            system.extend(_system_text(message.content, tlog))

        for block in _blocks(message.content):
            if block.type != "mid_conv_system":
                continue
            if control := block.cache_control:
                system.cache_controls.append(control)
            system.extend(_system_text(block.content, tlog))

    return system


def _convert_user_message(
    content: str | list[ContentBlock], tlog: TranslationLog
) -> list[SdkMessage]:
    if isinstance(content, str):
        if not content:
            return []
        return [SdkMessage(role=Role.USER, content=content)]

    tool_messages: list[SdkMessage] = []
    parts: list[MessageContentPart] = []

    for block in _blocks(content):
        match block.type:
            case "tool_result":
                tool_messages.append(_tool_result_message(block))
                parts.extend(_tool_result_images(block.content, tlog))
            case "text":
                if block.text:
                    parts.append(
                        MessageContentTextPart(type="text", text=block.text)
                    )
            case "image":
                if part := _image_part(block, tlog):
                    parts.append(part)
            case "document":
                if part := _document_part(block, tlog):
                    parts.append(part)
            case "mid_conv_system":
                continue
            case unsupported:
                tlog.warning(
                    "Dropping unsupported user content block: %s", unsupported
                )

    if parts:
        tool_messages.append(SdkMessage(role=Role.USER, content=parts))
    return tool_messages


def _tool_result_message(block: ContentBlock) -> SdkMessage:
    text: str = _tool_result_text(block.content)
    return SdkMessage(
        role=Role.TOOL,
        tool_call_id=block.tool_use_id,
        content=f"Error: {text}" if block.is_error else text,
    )


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        text
        for sub in _blocks(content)
        if sub.type == "text" and (text := sub.text)
    )


def _tool_result_images(
    content: object, tlog: TranslationLog
) -> list[MessageContentPart]:
    return [
        part
        for sub in _blocks(content)
        if sub.type == "image" and (part := _image_part(sub, tlog))
    ]


def _convert_assistant_message(
    content: str | list[ContentBlock],
    aliases: ToolNameAliases,
    tlog: TranslationLog,
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
        match block.type:
            case "text":
                if block.text:
                    text_parts.append(block.text)
            case "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id or "",
                        type="function",
                        function=FunctionCall(
                            name=aliases.to_upstream(block.name or ""),
                            arguments=json.dumps(block.input or {}),
                        ),
                    )
                )
            case "thinking" | "redacted_thinking" | "mid_conv_system":
                continue
            case unsupported:
                tlog.warning(
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


def _image_part(
    block: ContentBlock, tlog: TranslationLog
) -> MessageContentPart | None:
    source: ContentSource = block.source or ContentSource()
    stype: str | None = source.type
    if stype == "base64":
        media_type: str = source.media_type or "image/png"
        data: str = source.data or ""
        return MessageContentImagePart(
            type="image_url",
            image_url=ImageURL(url=f"data:{media_type};base64,{data}"),
        )
    if stype == "url" and (url := source.url):
        return MessageContentImagePart(
            type="image_url", image_url=ImageURL(url=url)
        )
    tlog.warning("Dropping image block with source type: %s", stype)
    return None


def _document_part(
    block: ContentBlock, tlog: TranslationLog
) -> MessageContentPart | None:
    source: ContentSource = block.source or ContentSource()
    stype: str | None = source.type
    filename: str = block.title or "document.pdf"
    if stype == "base64":
        media_type: str = source.media_type or "application/pdf"
        data: str = source.data or ""
        return MessageContentFilePart(
            type="file",
            file=InputFile(
                filename=filename,
                file_data=f"data:{media_type};base64,{data}",
            ),
        )
    if stype == "text" and (text_data := source.data):
        return MessageContentTextPart(type="text", text=text_data)

    tlog.warning("Dropping document block with source type: %s", stype)
    return None


def _convert_tools(
    tools: list[Tool] | None,
    aliases: ToolNameAliases,
    tlog: TranslationLog,
) -> list[SdkTool | StaticTool]:
    if not tools:
        return []
    result: list[SdkTool | StaticTool] = []
    for tool in tools:
        ttype: str | None = tool.type
        if ttype and ttype != "custom":
            tlog.warning("Dropping unsupported tool type: %s", ttype)
        elif not tool.name:
            tlog.warning("Dropping custom tool without a name")
        else:
            sdk_tool: SdkTool = SdkTool(
                type="function",
                function=SdkFunction(
                    name=aliases.to_upstream(tool.name),
                    description=tool.description or None,
                    parameters=_parameters(tool),
                    strict=False,
                ),
            )
            _mark_tool(
                sdk_tool,
                [tool.cache_control] if tool.cache_control else [],
                tlog,
            )
            result.append(sdk_tool)
    return result


def _parameters(tool: Tool) -> JsonObject:
    if tool.input_schema is None:
        return {"type": "object", "properties": {}}

    return {
        key: value
        for key, value in tool.input_schema.items()
        if key != "$schema"
    }


def _convert_tool_choice(
    tool_choice: ToolChoice | None, aliases: ToolNameAliases
) -> Literal["auto", "none", "required"] | SdkToolChoice | None:
    match tool_choice.type if tool_choice else None:
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
                    name=aliases.to_upstream(tool_choice.name or "")
                    if tool_choice
                    else ""
                ),
            )
        case _:
            return None


def _convert_parallel_tool_calls(tool_choice: ToolChoice | None) -> bool | None:
    if (
        not tool_choice
        or "disable_parallel_tool_use" not in tool_choice.model_fields_set
    ):
        return None
    return not bool(tool_choice.disable_parallel_tool_use)
