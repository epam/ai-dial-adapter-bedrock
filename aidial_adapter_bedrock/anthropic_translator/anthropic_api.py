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

from pydantic import field_validator

from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel

ContentBlock = dict[str, Any]


class Message(ExtraAllowModel):
    role: str
    content: str | list[ContentBlock]


class Tool(ExtraAllowModel):
    # Custom tools carry no `type`, or `"custom"`; server tools carry a
    # versioned one like "web_search_20250305".
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    type: str | None = None
    cache_control: dict[str, Any] | None = None


class ThinkingConfig(ExtraAllowModel):
    type: str | None = None
    budget_tokens: int | None = None

    @field_validator("budget_tokens", mode="before")
    @classmethod
    def _drop_bool(cls, value: Any) -> Any:
        # `bool` is an `int` subclass and pydantic coerces `True` to 1, which
        # the effort ladder would then read as a real budget.
        return None if isinstance(value, bool) else value


class OutputConfig(ExtraAllowModel):
    # Loosely typed so an unexpected value reaches the converter to be mapped
    # or dropped, rather than being rejected at validation.
    effort: str | None = None
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
    service_tier: str | None = None
    stream: bool | None = None
    # Declared only so the converters can warn-drop them by attribute access
    # rather than an untyped `getattr`; never forwarded.
    mcp_servers: list[Any] | None = None
    container: Any | None = None
    inference_geo: Any | None = None
    context_management: Any | None = None
    cache_control: dict[str, Any] | None = None
