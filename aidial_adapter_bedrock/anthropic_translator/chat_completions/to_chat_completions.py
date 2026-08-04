"""
Anthropic Messages request → OpenAI Chat Completions request body.

Unsupported input is dropped with a log line rather than rejected: Anthropic
clients send fields a given upstream cannot honour on every request, and
rejecting them makes the client unusable. Only a missing `max_tokens` and an
unrecognised message `role` fail a request.

Which shape the outbound body takes is decided per deployment from its
capability profile — the reasoning knob, the output-cap field name, whether
`temperature` survives, and whether prompt-cache markers mean anything. Model
and adapter-specific knobs outside the standard schema travel under
`custom_fields.configuration`, never at the top level, because strict adapters
reject unrecognised top-level fields.
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
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    DeploymentProfile,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    resolve_reasoning,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    INVALID_REQUEST_ERROR,
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.stop_sequences import (
    strips_stop_parameter,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)
from aidial_adapter_bedrock.anthropic_translator.translation_log import (
    TranslationLog,
)

ContentBlock = dict[str, Any]

# `metadata.user_id` is an abuse-detection identifier; a silently shortened one
# is worse than none, so an over-long value is dropped rather than truncated.
_MAX_USER_LENGTH = 64

# OpenAI's `response_format.json_schema.name` is required; Anthropic's
# `output_config.format` has no per-schema name to carry over, so a fixed
# placeholder is used for every request.
_JSON_SCHEMA_NAME = "response"

# Anthropic's `service_tier` is a closed enum, so an unrecognised value is
# dropped rather than forwarded.
_SERVICE_TIERS = {"auto": "auto", "standard_only": "default"}


class CoreChatCompletionRequest(ChatCompletionRequest):
    """The aidial-sdk request model plus the standard Chat Completions fields
    it doesn't declare."""

    service_tier: str | None = None


def to_chat_completions_request(
    req: MessagesRequest, deployment: str, profile: DeploymentProfile
) -> tuple[CoreChatCompletionRequest, ToolNameAliases]:
    """The outbound body, and the tool-name aliases its response must reverse."""
    tlog = TranslationLog("Anthropic→Chat Completions request")
    aliases = ToolNameAliases()
    try:
        if req.max_tokens is None:
            raise AnthropicHTTPError(
                400, INVALID_REQUEST_ERROR, "'max_tokens' is required"
            )

        messages: list[SdkMessage] = []
        cache = profile.cache_supported

        system_texts, system_cached = _collect_system(req)
        if system_texts:
            system_message = SdkMessage(
                role=Role.SYSTEM, content="\n\n".join(system_texts)
            )
            if cache and system_cached:
                _mark_message_cache_breakpoint(system_message)
            messages.append(system_message)

        for message in req.messages:
            match message.role:
                case "user":
                    converted = _convert_user_message(message.content, tlog)
                case "assistant":
                    converted = _convert_assistant_message(
                        message.content, aliases, tlog
                    )
                case "system":
                    continue  # merged into the leading system message above
                case unknown:
                    raise AnthropicHTTPError(
                        400,
                        INVALID_REQUEST_ERROR,
                        f"Unknown message role: {unknown!r}",
                    )
            if cache and _has_cache_control(message.content):
                for converted_message in converted:
                    _mark_message_cache_breakpoint(converted_message)
            messages.extend(converted)

        tool_choice, parallel_tool_calls = _convert_tool_choice(
            req.tool_choice, aliases
        )

        reasoning_effort, configuration = resolve_reasoning(req, profile, tlog)
        if _any_citations_enabled(req.messages):
            configuration["enable_citations"] = True

        _warn_dropped(req, tlog)
        max_tokens = _cap_output(req.max_tokens, profile, tlog)
        uses_new_spelling = profile.max_completion_tokens_supported

        return (
            CoreChatCompletionRequest(
                model=deployment,
                messages=messages,
                custom_fields=(
                    ChatCompletionRequestCustomFields(
                        configuration=configuration
                    )
                    if configuration
                    else None
                ),
                tools=_convert_tools(req.tools, cache, aliases, tlog) or None,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                reasoning_effort=reasoning_effort,
                response_format=_convert_response_format(req, tlog),
                stop=_convert_stop(req, deployment, tlog),
                max_tokens=None if uses_new_spelling else max_tokens,
                max_completion_tokens=max_tokens if uses_new_spelling else None,
                temperature=(
                    req.temperature if profile.temperature_supported else None
                ),
                top_p=req.top_p,
                user=_convert_user(req, tlog),
                service_tier=_convert_service_tier(req, tlog),
            ),
            aliases,
        )
    finally:
        tlog.flush()


def _warn_dropped(req: MessagesRequest, tlog: TranslationLog) -> None:
    if req.top_k is not None:
        tlog.debug("Dropping unsupported 'top_k' parameter")
    if req.mcp_servers:
        tlog.warning("Dropping 'mcp_servers': no Chat Completions equivalent")
    if req.container:
        tlog.warning("Dropping 'container': no Chat Completions equivalent")
    if req.inference_geo:
        tlog.warning("Dropping 'inference_geo': no Chat Completions equivalent")
    if req.cache_control:
        # "Mark the last cacheable block" has no faithful translation once
        # blocks are flattened into messages, so it is dropped, not guessed at.
        tlog.warning("Dropping top-level 'cache_control'")


def _cap_output(
    max_tokens: int, profile: DeploymentProfile, tlog: TranslationLog
) -> int:
    """Clamp the client's cap down to the deployment's ceiling, never upward."""
    ceiling = profile.max_output_tokens
    if ceiling is None or max_tokens <= ceiling:
        return max_tokens
    tlog.debug(
        "Clamping max_tokens %d to the deployment's %d", max_tokens, ceiling
    )
    return ceiling


def _convert_stop(
    req: MessagesRequest, deployment: str, tlog: TranslationLog
) -> list[str] | None:
    if not req.stop_sequences:
        return None
    if strips_stop_parameter(deployment):
        # The deployment rejects the parameter outright, so the sequences are
        # reproduced on the response path instead.
        tlog.debug("Omitting 'stop': %s rejects the parameter", deployment)
        return None
    return req.stop_sequences


def _convert_user(req: MessagesRequest, tlog: TranslationLog) -> str | None:
    user_id = req.metadata.user_id if req.metadata else None
    if not user_id:
        return None
    if len(user_id) > _MAX_USER_LENGTH:
        tlog.debug(
            "Dropping 'metadata.user_id': longer than %d characters",
            _MAX_USER_LENGTH,
        )
        return None
    return user_id


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
            # Anthropic's structured output always guarantees schema adherence
            # and has no non-strict mode, so `true` is the faithful reading.
            strict=True,
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
        for block in _blocks(message.content):
            if block.get("type") == "document" and (
                block.get("citations") or {}
            ).get("enabled"):
                return True
    return False


def _blocks(content: Any) -> list[ContentBlock]:
    return content if isinstance(content, list) else []


def _system_text(content: Any) -> tuple[list[str], bool]:
    """The text a system-shaped value carries, and whether any block of it is a
    cache breakpoint."""
    if isinstance(content, str):
        return ([content] if content else []), False

    texts: list[str] = []
    cached = False
    for block in _blocks(content):
        cached = cached or bool(block.get("cache_control"))
        if block.get("type") == "text" and (text := block.get("text")):
            texts.append(text)
    return texts, cached


def _collect_system(req: MessagesRequest) -> tuple[list[str], bool]:
    """Every system-origin text, in the order it must be joined, and whether
    any of it carried a cache breakpoint.

    Some Chat Completions adapters reject more than one system message, so the
    top-level `system` field, mid-conversation `system`-role turns real clients
    like Claude Code send, and `mid_conv_system` blocks on a message of any role
    all merge into one leading system message.
    """
    texts, cached = _system_text(req.system)
    role_texts: list[str] = []
    nested_texts: list[str] = []

    for message in req.messages:
        if message.role == "system":
            message_texts, message_cached = _system_text(message.content)
            role_texts.extend(message_texts)
            cached = cached or message_cached

        for block in _blocks(message.content):
            if block.get("type") != "mid_conv_system":
                continue
            cached = cached or bool(block.get("cache_control"))
            block_texts, block_cached = _system_text(block.get("content"))
            nested_texts.extend(block_texts)
            cached = cached or block_cached

    return texts + role_texts + nested_texts, cached


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
        elif btype == "mid_conv_system":
            continue  # merged into the leading system message
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
                        name=aliases.to_upstream(block.get("name") or ""),
                        arguments=json.dumps(block.get("input") or {}),
                    ),
                )
            )
        elif btype in ("thinking", "redacted_thinking"):
            # Signatures are provider-specific and unverifiable downstream;
            # there is no replay path through this dialect.
            continue
        elif btype == "mid_conv_system":
            continue  # merged into the leading system message
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
    tools: list[Tool] | None,
    cache: bool,
    aliases: ToolNameAliases,
    tlog: TranslationLog,
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
            # Completions equivalent at all — dropped, not rejected. They carry
            # no `input_schema`, so forcing them through the function shape
            # would produce a malformed definition.
            tlog.warning("Dropping unsupported tool type: %s", ttype)
        elif not tool.name:
            tlog.warning("Dropping custom tool without a name")
        else:
            sdk_tool = SdkTool(
                type="function",
                function=SdkFunction(
                    name=aliases.to_upstream(tool.name),
                    description=tool.description,
                    parameters=_parameters(tool),
                    # `strict: False` is deliberate: Anthropic tool schemas
                    # routinely fail OpenAI's strict-mode requirements, which
                    # would reject working tool definitions.
                    strict=False,
                ),
            )
            if cache and tool.cache_control:
                _mark_tool_cache_breakpoint(sdk_tool)
            result.append(sdk_tool)
    return result


def _parameters(tool: Tool) -> dict[str, Any]:
    if tool.input_schema is None:
        return {"type": "object", "properties": {}}
    # `$schema` is legal JSON Schema but not part of the subset this field
    # allows, and strict adapters reject the whole request over it.
    return {
        key: value
        for key, value in tool.input_schema.items()
        if key != "$schema"
    }


def _convert_tool_choice(
    tool_choice: dict[str, Any] | None, aliases: ToolNameAliases
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
                function=FunctionChoice(
                    name=aliases.to_upstream(tool_choice.get("name") or "")
                ),
            )
        case _:
            choice = None

    return choice, parallel
