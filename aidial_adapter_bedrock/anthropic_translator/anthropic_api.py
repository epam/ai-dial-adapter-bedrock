"""
Tolerant pydantic models for the incoming Anthropic Messages request body.

Tolerance beats strictness here: Claude Code and the Anthropic SDKs send fields
this translator does not use, and future fields we have never seen. Every model
extends `ExtraAllowModel` so unknown keys survive validation (visible for
logging, never forwarded). Only the fields the translator relies on are typed;
content blocks stay as raw dicts and are interpreted defensively by the request
converters.
"""

from typing import Any

from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel

# A content block is a heterogeneous, provider-defined object. Keeping it as a
# dict (rather than a strict union) is intentional — see module docstring.
ContentBlock = dict[str, Any]


class Message(ExtraAllowModel):
    role: str
    content: str | list[ContentBlock]


class Tool(ExtraAllowModel):
    # Custom tools carry `name` + `input_schema` (and either no `type` or
    # `type == "custom"`). Server tools (web_search, bash, text_editor, …)
    # carry a versioned `type` like "web_search_20250305".
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    type: str | None = None


class ThinkingConfig(ExtraAllowModel):
    type: str | None = None
    budget_tokens: int | None = None


class OutputConfig(ExtraAllowModel):
    # Anthropic effort levels are `low | medium | high | max`. Typed as a plain
    # string (like `ThinkingConfig.type`) so an unexpected value survives to be
    # mapped or dropped by the converter rather than rejected at validation.
    effort: str | None = None
    # Structured-output constraint, e.g. `{"type": "json_schema", "schema":
    # {...}}`. Kept as a raw dict (like `Tool.input_schema`) so the converter
    # can validate/drop it defensively rather than rejecting at validation.
    format: dict[str, Any] | None = None


class Metadata(ExtraAllowModel):
    user_id: str | None = None


class MessagesRequest(ExtraAllowModel):
    """Union of the Messages create body and the count_tokens body.

    `max_tokens` and `stream` are required only for the create endpoint; the
    handler enforces `max_tokens` there and this model keeps it optional so the
    same converter serves count_tokens.
    """

    model: str | None = None
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[ContentBlock] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None
    output_config: OutputConfig | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: Metadata | None = None
    stream: bool | None = None
    # Declared only so the converters can detect and warn-drop them with a
    # typed attribute access rather than an untyped `getattr`; never forwarded.
    mcp_servers: list[Any] | None = None
    container: Any | None = None
