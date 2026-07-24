"""
Anthropic Messages request → OpenAI Chat Completions request body.

Model/adapter-specific knobs outside the standard schema (e.g. citation
toggles) travel under `custom_fields.configuration`, emitted only when that
container is non-empty. Reasoning effort and structured output are the
exceptions: they always map to the standard `reasoning_effort` and
`response_format` fields (never nested under `custom_fields`), resolved from
`output_config.effort` and `output_config.format` respectively. Deployments
that don't support these fields will reject them — a capability gap this
translator can't paper over generically.

Anthropic prompt-cache breakpoints (`cache_control` on a content block or
tool) have no OpenAI-schema counterpart, but DIAL Core has its own,
per-message/per-tool `custom_fields.cache_breakpoint` marker
(https://docs.dialx.ai/tutorials/developers/prompt-caching). Presence of
`cache_control` anywhere in a turn's content unconditionally sets that marker
on every Chat-Completions message the turn produces — no model-catalog
capability check.
"""

import json
from typing import Any, Literal

from aidial_sdk.chat_completion.request import (
    CacheBreakpoint,
    ChatCompletionRequest,
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
    ReasoningEffort,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaObject,
    Role,
    StaticTool,
    ToolCall,
    ToolCustomFields,
)
from aidial_sdk.chat_completion.request import (
    Function as SdkFunction,
)
from aidial_sdk.chat_completion.request import (
    Message as SdkMessage,
)
from aidial_sdk.chat_completion.request import (
    Tool as SdkTool,
)
from aidial_sdk.chat_completion.request import (
    ToolChoice as SdkToolChoice,
)

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    Message,
    MessagesRequest,
    Tool,
)
from aidial_adapter_bedrock.anthropic_translator.common import (
    resolve_effort,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    INVALID_REQUEST_ERROR,
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

ContentBlock = dict[str, Any]


def to_chat_completions_request(
    req: MessagesRequest, model: str
) -> ChatCompletionRequest:
    tlog = TranslationLog("Anthropic→Chat Completions request")
    try:
        messages: list[SdkMessage] = []

        # Some Chat Completions adapters 400 on more than one system message,
        # so every system-origin text — the top-level `system` field plus any
        # mid-conversation `system`-role turns real clients like Claude Code
        # send — is merged into one leading system message.
        system_texts: list[str] = []
        system_cached = False
        if (top_level_system := _convert_system(req.system)) is not None:
            system_texts.append(top_level_system)
            system_cached = system_cached or _has_cache_control(req.system)
        for message in req.messages:
            if message.role == "system" and (
                text := _convert_system_role_text(message.content, tlog)
            ):
                system_texts.append(text)
                system_cached = system_cached or _has_cache_control(
                    message.content
                )
        if system_texts:
            system_message = SdkMessage(
                role=Role.SYSTEM, content="\n\n".join(system_texts)
            )
            if system_cached:
                _mark_message_cache_breakpoint(system_message)
            messages.append(system_message)

        for message in req.messages:
            if message.role == "user":
                converted = _convert_user_message(message.content, tlog)
                if _has_cache_control(message.content):
                    for converted_message in converted:
                        _mark_message_cache_breakpoint(converted_message)
                messages.extend(converted)
            elif message.role == "assistant":
                converted = _convert_assistant_message(message.content, tlog)
                if _has_cache_control(message.content):
                    for converted_message in converted:
                        _mark_message_cache_breakpoint(converted_message)
                messages.extend(converted)
            elif message.role == "system":
                continue  # merged into the leading system message above
            else:
                raise AnthropicHTTPError(
                    400,
                    INVALID_REQUEST_ERROR,
                    f"Unknown message role: {message.role!r}",
                )

        tool_choice, parallel_tool_calls = _convert_tool_choice(req.tool_choice)

        if req.top_k is not None:
            tlog.debug("Dropping unsupported 'top_k' parameter")
        if req.mcp_servers:
            tlog.warning(
                "Dropping 'mcp_servers': no Chat Completions equivalent"
            )
        if req.container:
            tlog.warning("Dropping 'container': no Chat Completions equivalent")

        # `metadata.user_id` is intentionally NOT forwarded as
        # `safety_identifier`: the vertexai-adapter drops it downstream, so
        # support would be inconsistent across deployments.
        return ChatCompletionRequest(
            model=model,
            messages=messages,
            custom_fields=_convert_custom_fields(req),
            tools=_convert_tools(req.tools, tlog) or None,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=_convert_reasoning_effort(req, tlog),
            response_format=_convert_response_format(req, tlog),
            stop=req.stop_sequences or None,
            max_completion_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
    finally:
        tlog.flush()


def _convert_custom_fields(
    req: MessagesRequest,
) -> ChatCompletionRequestCustomFields | None:
    configuration: dict[str, Any] = {}
    if _any_citations_enabled(req.messages):
        configuration["enable_citations"] = True
    return (
        ChatCompletionRequestCustomFields(configuration=configuration)
        if configuration
        else None
    )


def _convert_reasoning_effort(
    req: MessagesRequest, tlog: TranslationLog
) -> ReasoningEffort | None:
    effort = resolve_effort(req.output_config)
    if effort is None:
        return None
    if effort == "xhigh":
        # DIAL's Chat Completions `ReasoningEffort` has no `xhigh` (unlike the
        # Responses API); clamp to the highest level it supports.
        tlog.debug("Clamping reasoning effort 'xhigh' to 'high'")
        effort = "high"
    return ReasoningEffort(effort)


# OpenAI's `response_format.json_schema.name` is required; Anthropic's
# `output_config.format` has no per-schema name to carry over, so a fixed
# placeholder is used for every request.
_JSON_SCHEMA_NAME = "response"


def _convert_response_format(
    req: MessagesRequest, tlog: TranslationLog
) -> ResponseFormat | None:
    output_format = req.output_config.format if req.output_config else None
    if output_format is None:
        return None
    if output_format.get("type") != "json_schema":
        tlog.warning(
            "Dropping unsupported output_config.format type: %s",
            output_format.get("type"),
        )
        return None
    schema = output_format.get("schema")
    if not isinstance(schema, dict):
        tlog.warning("Dropping output_config.format: missing 'schema'")
        return None
    return ResponseFormatJsonSchema(
        type="json_schema",
        json_schema=ResponseFormatJsonSchemaObject(
            name=_JSON_SCHEMA_NAME,
            schema=schema,
            # `strict: False` is deliberate, mirroring `_convert_tools`:
            # Anthropic's `additionalProperties: false` requirement is close
            # to but not identical with OpenAI's strict-mode constraints.
            strict=False,
        ),
    )


def _has_cache_control(content: str | list[ContentBlock] | None) -> bool:
    return isinstance(content, list) and any(
        block.get("cache_control") for block in content
    )


def _mark_message_cache_breakpoint(message: SdkMessage) -> None:
    message.custom_fields = MessageCustomFields(
        cache_breakpoint=CacheBreakpoint()
    )


def _mark_tool_cache_breakpoint(tool: SdkTool) -> None:
    tool.custom_fields = ToolCustomFields(cache_breakpoint=CacheBreakpoint())


def _any_citations_enabled(messages: list[Message]) -> bool:
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for block in message.content:
            if block.get("type") == "document" and (
                block.get("citations") or {}
            ).get("enabled"):
                return True
    return False


def _convert_system(system: str | list[ContentBlock] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    texts: list[str] = []
    for block in system:
        if block.get("type") == "text" and (text := block.get("text")):
            texts.append(text)
    return "\n\n".join(texts) if texts else None


def _convert_system_role_text(
    content: str | list[ContentBlock], tlog: TranslationLog
) -> str | None:
    if isinstance(content, str):
        return content or None

    texts: list[str] = []
    for block in content:
        if block.get("type") == "text" and (text := block.get("text")):
            texts.append(text)
        else:
            tlog.warning(
                "Dropping unsupported system content block: %s",
                block.get("type"),
            )
    return "\n\n".join(texts) if texts else None


def _convert_user_message(
    content: str | list[ContentBlock], tlog: TranslationLog
) -> list[SdkMessage]:
    if isinstance(content, str):
        if not content:
            return []
        return [SdkMessage(role=Role.USER, content=content)]

    # tool_result blocks become top-level `tool` messages that must precede
    # the residual user message (they answer the preceding tool calls).
    tool_messages: list[SdkMessage] = []
    parts: list[MessageContentPart] = []

    for block in content:
        btype: str | None = block.get("type")
        if btype == "tool_result":
            text, image_parts = _tool_result_output(block, tlog)
            if block.get("is_error"):
                text = f"Error: {text}"
            tool_messages.append(
                SdkMessage(
                    role=Role.TOOL,
                    tool_call_id=block.get("tool_use_id"),
                    content=text,
                )
            )
            # Images can't ride inside a tool message, so surface them as
            # image_url parts of the residual user message.
            parts.extend(image_parts)
        elif btype == "text":
            if text := block.get("text"):
                parts.append(MessageContentTextPart(type="text", text=text))
        elif btype == "image":
            if part := _image_part(block, tlog):
                parts.append(part)
        elif btype == "document":
            if part := _document_part(block, tlog):
                parts.append(part)
        else:
            tlog.warning("Dropping unsupported user content block: %s", btype)

    items = list(tool_messages)
    if parts:
        items.append(SdkMessage(role=Role.USER, content=parts))
    return items


def _tool_result_output(
    block: ContentBlock, tlog: TranslationLog
) -> tuple[str, list[MessageContentPart]]:
    content: Any = block.get("content")
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []

    texts: list[str] = []
    images: list[MessageContentPart] = []
    for sub in content:
        if not isinstance(sub, dict):
            continue
        if sub.get("type") == "text":
            if text := sub.get("text"):
                texts.append(text)
        elif sub.get("type") == "image" and (part := _image_part(sub, tlog)):
            images.append(part)
    return "\n".join(texts), images


def _convert_assistant_message(
    content: str | list[ContentBlock], tlog: TranslationLog
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
        btype: str | None = block.get("type")
        if btype == "text":
            if text := block.get("text"):
                text_parts.append(text)
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.get("id") or "",
                    type="function",
                    function=FunctionCall(
                        name=block.get("name") or "",
                        arguments=json.dumps(block.get("input") or {}),
                    ),
                )
            )
        elif btype in ("thinking", "redacted_thinking"):
            continue
        else:
            tlog.warning(
                "Dropping unsupported assistant content block: %s", btype
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
    source = block.get("source") or {}
    stype: str | None = source.get("type")
    if stype == "base64":
        media_type: str = source.get("media_type") or "image/png"
        data: str = source.get("data") or ""
        return MessageContentImagePart(
            type="image_url",
            image_url=ImageURL(url=f"data:{media_type};base64,{data}"),
        )
    if stype == "url" and (url := source.get("url")):
        return MessageContentImagePart(
            type="image_url", image_url=ImageURL(url=url)
        )
    tlog.warning("Dropping image block with source type: %s", stype)
    return None


def _document_part(
    block: ContentBlock, tlog: TranslationLog
) -> MessageContentPart | None:
    source = block.get("source") or {}
    stype: str | None = source.get("type")
    filename: str = block.get("title") or "document.pdf"
    if stype == "base64":
        media_type: str = source.get("media_type") or "application/pdf"
        data: str = source.get("data") or ""
        return MessageContentFilePart(
            type="file",
            file=InputFile(
                filename=filename,
                file_data=f"data:{media_type};base64,{data}",
            ),
        )
    if stype == "text" and (text_data := source.get("data")):
        return MessageContentTextPart(type="text", text=text_data)
    # Chat Completions' `file` part has no remote-URL form (only inline
    # `file_data` or a previously-uploaded `file_id`), unlike Responses.
    tlog.warning("Dropping document block with source type: %s", stype)
    return None


def _convert_tools(
    tools: list[Tool] | None, tlog: TranslationLog
) -> list[SdkTool | StaticTool]:
    # Annotated as `SdkTool | StaticTool` (never `StaticTool` in practice) to
    # match `ChatCompletionRequest.tools`'s declared type: `list` is
    # invariant, so a bare `list[SdkTool]` isn't assignable to it.
    if not tools:
        return []
    result: list[SdkTool | StaticTool] = []
    for tool in tools:
        ttype: str | None = tool.type
        if ttype and ttype != "custom":
            # Server tools (web_search, bash, text_editor, …) have no Chat
            # Completions equivalent at all — dropped, not rejected.
            tlog.warning("Dropping unsupported tool type: %s", ttype)
        elif not tool.name:
            tlog.warning("Dropping custom tool without a name")
        else:
            sdk_tool = SdkTool(
                type="function",
                function=SdkFunction(
                    name=tool.name,
                    description=tool.description,
                    parameters=(
                        tool.input_schema
                        if tool.input_schema is not None
                        else {"type": "object", "properties": {}}
                    ),
                    # `strict: False` is deliberate: Anthropic tool
                    # schemas routinely fail OpenAI's strict-mode
                    # requirements, which would reject them.
                    strict=False,
                ),
            )
            if getattr(tool, "cache_control", None):
                _mark_tool_cache_breakpoint(sdk_tool)
            result.append(sdk_tool)
    return result


def _convert_tool_choice(
    tool_choice: dict[str, Any] | None,
) -> tuple[
    Literal["auto", "none", "required"] | SdkToolChoice | None, bool | None
]:
    if not tool_choice:
        return None, None

    parallel: bool | None = None
    if "disable_parallel_tool_use" in tool_choice:
        parallel = not bool(tool_choice.get("disable_parallel_tool_use"))

    choice: Literal["auto", "none", "required"] | SdkToolChoice | None
    match tool_choice.get("type"):
        case "auto":
            choice = "auto"
        case "any":
            choice = "required"
        case "none":
            choice = "none"
        case "tool":
            # `name` is guaranteed by Anthropic's `{"type": "tool", "name":
            # ...}` shape; `or ""` is a defensive fallback only.
            choice = SdkToolChoice(
                type="function",
                function=FunctionChoice(name=tool_choice.get("name") or ""),
            )
        case _:
            choice = None

    return choice, parallel
