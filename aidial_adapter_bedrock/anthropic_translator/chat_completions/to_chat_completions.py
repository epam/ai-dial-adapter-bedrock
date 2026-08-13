"""
Anthropic Messages request → OpenAI Chat Completions request body.

Unsupported input is dropped with a log line rather than rejected: Anthropic
clients send fields a given upstream cannot honour on every request, and
rejecting them makes the client unusable. Only a missing `max_tokens` and an
unrecognised message `role` fail a request.

Which shape the outbound body takes is decided per deployment from its
capability profile — the reasoning effort, the output-cap field name, whether
`temperature` survives, and whether prompt-cache markers mean anything.
"""

import json
from typing import Any, Literal

from aidial_sdk.chat_completion.request import (
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
from pydantic import BaseModel

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    Message,
    MessagesRequest,
    Tool,
)
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    DeploymentProfile,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.cache_breakpoints import (
    CacheControl,
    cache_breakpoint,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.reasoning import (
    resolve_reasoning_effort,
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

_MAX_USER_LENGTH = 64

# `output_config.format` has no per-schema name to carry over, and OpenAI
# requires one.
_JSON_SCHEMA_NAME = "response"

_SERVICE_TIERS = {"auto": "auto", "standard_only": "default"}


class CoreChatCompletionRequest(ChatCompletionRequest):
    """The aidial-sdk request model plus the standard Chat Completions fields
    it doesn't declare."""

    # The SDK's `ReasoningEffort` enum stops at `high`; DIAL deployments also
    # advertise `minimum`, `xhigh` and `max`, and the gate only ever emits a
    # level the deployment itself named.
    reasoning_effort: str | None = None
    service_tier: str | None = None


class SystemPrompt(BaseModel):
    """The system-origin text destined for the one leading system message, and
    the `cache_control` objects that travelled with it."""

    texts: list[str] = []
    cache_controls: list[CacheControl] = []

    @property
    def text(self) -> str:
        return "\n\n".join(self.texts)

    def extend(self, other: "SystemPrompt") -> None:
        self.texts.extend(other.texts)
        self.cache_controls.extend(other.cache_controls)


def to_chat_completions_request(
    req: MessagesRequest,
    deployment: str,
    profile: DeploymentProfile,
    aliases: ToolNameAliases,
) -> CoreChatCompletionRequest:
    """The outbound body. Every tool name it had to alias is registered in
    `aliases`, which the response path reverses."""
    tlog = TranslationLog("Anthropic→Chat Completions request")
    try:
        if req.max_tokens is None:
            raise AnthropicHTTPError(
                400, INVALID_REQUEST_ERROR, "'max_tokens' is required"
            )

        cache: bool = profile.cache_supported
        messages: list[SdkMessage] = _convert_messages(
            req, cache, aliases, tlog
        )
        configuration: dict[str, Any] = _configuration(req)
        uses_new_spelling: bool = profile.max_completion_tokens_supported

        _warn_dropped(req, tlog)

        return CoreChatCompletionRequest(
            model=deployment,
            messages=messages,
            custom_fields=(
                ChatCompletionRequestCustomFields(configuration=configuration)
                if configuration
                else None
            ),
            tools=_convert_tools(req.tools, cache, aliases, tlog) or None,
            tool_choice=_convert_tool_choice(req.tool_choice, aliases),
            parallel_tool_calls=_convert_parallel_tool_calls(req.tool_choice),
            reasoning_effort=resolve_reasoning_effort(req, profile, tlog),
            response_format=_convert_response_format(req, tlog),
            stop=_convert_stop(req, deployment, tlog),
            # Forwarded verbatim: the features header carries no limits, so
            # there is no deployment ceiling to clamp against.
            max_tokens=None if uses_new_spelling else req.max_tokens,
            max_completion_tokens=(
                req.max_tokens if uses_new_spelling else None
            ),
            temperature=(
                req.temperature if profile.temperature_supported else None
            ),
            top_p=req.top_p,
            user=_convert_user(req, tlog),
            service_tier=_convert_service_tier(req, tlog),
        )
    finally:
        tlog.flush()


def _convert_messages(
    req: MessagesRequest,
    cache: bool,
    aliases: ToolNameAliases,
    tlog: TranslationLog,
) -> list[SdkMessage]:
    messages: list[SdkMessage] = []

    system: SystemPrompt = _collect_system(req, tlog)
    if system.texts:
        system_message = SdkMessage(role=Role.SYSTEM, content=system.text)
        _mark_message(system_message, cache, system.cache_controls, tlog)
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
        # One turn can split into several messages — a `tool_result` turn
        # becomes a `tool` message plus a residual `user` one — and every one
        # of them carries the turn's cache marker.
        controls: list[CacheControl] = _cache_controls(message.content)
        for converted_message in converted:
            _mark_message(converted_message, cache, controls, tlog)
        messages.extend(converted)

    return messages


def _configuration(req: MessagesRequest) -> dict[str, Any]:
    """Adapter knobs outside the standard schema, which travel under
    `custom_fields.configuration` because strict adapters reject unrecognised
    top-level fields."""
    if _any_citations_enabled(req.messages):
        return {"enable_citations": True}
    return {}


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
    if req.cache_control:
        # "Mark the last cacheable block" has no faithful translation once
        # blocks are flattened into messages, so it is not approximated.
        tlog.warning("Dropping top-level 'cache_control'")


def _convert_stop(
    req: MessagesRequest, deployment: str, tlog: TranslationLog
) -> list[str] | None:
    if not req.stop_sequences:
        return None
    if strips_stop_parameter(deployment):
        # Reproduced on the response path instead — see `stop_sequences`.
        tlog.debug("Omitting 'stop': %s rejects the parameter", deployment)
        return None
    return req.stop_sequences


def _convert_user(req: MessagesRequest, tlog: TranslationLog) -> str | None:
    """An abuse-detection identifier: a silently shortened one is worse than
    none, so an over-long value is dropped rather than truncated."""
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
    """A closed enum, so an unrecognised value is dropped rather than
    forwarded."""
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


def _cache_controls(content: Any) -> list[CacheControl]:
    """The `cache_control` objects on a content array's **top-level** blocks.

    One nested inside a `tool_result`'s own content is not seen: marking is
    per-message here, and there is no object to hang a nested one on.
    """
    return [
        control
        for block in _blocks(content)
        if isinstance(control := block.get("cache_control"), dict)
    ]


def _mark_message(
    message: SdkMessage,
    cache: bool,
    controls: list[CacheControl],
    tlog: TranslationLog,
) -> None:
    if not cache:
        return
    if (marker := cache_breakpoint(controls, tlog)) is not None:
        message.custom_fields = MessageCustomFields(cache_breakpoint=marker)


def _mark_tool(
    tool: SdkTool,
    cache: bool,
    controls: list[CacheControl],
    tlog: TranslationLog,
) -> None:
    if not cache:
        return
    if (marker := cache_breakpoint(controls, tlog)) is not None:
        tool.custom_fields = ToolCustomFields(cache_breakpoint=marker)


def _any_citations_enabled(messages: list[Message]) -> bool:
    for message in messages:
        for block in _blocks(message.content):
            if block.get("type") == "document" and (
                block.get("citations") or {}
            ).get("enabled"):
                return True
    return False


def _blocks(content: Any) -> list[ContentBlock]:
    """A content array's blocks, read defensively: a non-object entry is not a
    block, and anything but an array carries none."""
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _system_text(content: Any, tlog: TranslationLog) -> SystemPrompt:
    """The text a system-shaped value carries, and the `cache_control` objects
    its blocks carry."""
    if isinstance(content, str):
        return SystemPrompt(texts=[content] if content else [])

    texts: list[str] = []
    for block in _blocks(content):
        match block.get("type"):
            case "text":
                if text := block.get("text"):
                    texts.append(text)
            case "mid_conv_system":
                continue  # collected by the caller, in conversation order
            case unsupported:
                tlog.warning(
                    "Dropping unsupported system content block: %s", unsupported
                )
    return SystemPrompt(texts=texts, cache_controls=_cache_controls(content))


def _collect_system(req: MessagesRequest, tlog: TranslationLog) -> SystemPrompt:
    """Every system-origin text, in the order it must be joined.

    Some Chat Completions adapters reject more than one system message, so the
    top-level `system` field, mid-conversation `system`-role turns real clients
    like Claude Code send, and `mid_conv_system` blocks on a message of any role
    all merge into one leading system message — in client order, not grouped by
    kind. All of their breakpoints therefore land on that one message.
    """
    system: SystemPrompt = _system_text(req.system, tlog)

    for message in req.messages:
        if message.role == "system":
            system.extend(_system_text(message.content, tlog))

        for block in _blocks(message.content):
            if block.get("type") != "mid_conv_system":
                continue
            if isinstance(control := block.get("cache_control"), dict):
                system.cache_controls.append(control)
            system.extend(_system_text(block.get("content"), tlog))

    return system


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

    for block in _blocks(content):
        btype: str | None = block.get("type")
        if btype == "tool_result":
            tool_messages.append(_tool_result_message(block))
            parts.extend(_tool_result_images(block.get("content"), tlog))
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


def _tool_result_message(block: ContentBlock) -> SdkMessage:
    """Read defensively: a missing `tool_use_id` emits `tool_call_id: null` and
    unreadable content emits `""`, leaving the rejection to the upstream."""
    text: str = _tool_result_text(block.get("content"))
    return SdkMessage(
        role=Role.TOOL,
        tool_call_id=block.get("tool_use_id"),
        content=f"Error: {text}" if block.get("is_error") else text,
    )


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        text
        for sub in _blocks(content)
        if sub.get("type") == "text" and (text := sub.get("text"))
    )


def _tool_result_images(
    content: Any, tlog: TranslationLog
) -> list[MessageContentPart]:
    """Images cannot ride inside a `tool` message, so they surface as
    `image_url` parts of the residual user message instead. Serialising the
    whole array to a JSON string would make tool-produced screenshots
    invisible to the model."""
    return [
        part
        for sub in _blocks(content)
        if sub.get("type") == "image" and (part := _image_part(sub, tlog))
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
            # No replay path through this dialect: signatures are
            # provider-specific and unverifiable downstream.
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
    # The `file` part has no remote-URL form, only inline `file_data` or a
    # previously-uploaded `file_id`.
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
            # Server tools (web_search, bash, text_editor, …) carry no
            # `input_schema`, so forcing them through the function shape would
            # produce a malformed definition.
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
            _mark_tool(
                sdk_tool,
                cache,
                [tool.cache_control] if tool.cache_control else [],
                tlog,
            )
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
) -> Literal["auto", "none", "required"] | SdkToolChoice | None:
    match (tool_choice or {}).get("type"):
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
                    name=aliases.to_upstream(tool_choice.get("name") or "")
                    if tool_choice
                    else ""
                ),
            )
        case _:
            return None


def _convert_parallel_tool_calls(
    tool_choice: dict[str, Any] | None,
) -> bool | None:
    """Read independently of `tool_choice.type`, so a choice carrying only this
    key emits `parallel_tool_calls` and no `tool_choice`."""
    if not tool_choice or "disable_parallel_tool_use" not in tool_choice:
        return None
    return not bool(tool_choice.get("disable_parallel_tool_use"))
