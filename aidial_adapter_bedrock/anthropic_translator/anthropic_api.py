from pydantic import Field, JsonValue, field_validator

from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel

JsonObject = dict[str, JsonValue]


class CacheControl(ExtraAllowModel):
    ttl: JsonValue = None

    def __bool__(self) -> bool:
        return bool(self.model_fields_set or self.model_extra)


class ContentSource(ExtraAllowModel):
    type: str | None = None
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class CitationsConfig(ExtraAllowModel):
    enabled: bool = False


class ContentBlock(ExtraAllowModel):
    type: str | None = None
    text: str | None = None
    source: ContentSource | None = None
    cache_control: CacheControl | None = None
    citations: CitationsConfig | None = None
    content: JsonValue = None
    tool_use_id: str | None = None
    is_error: bool = False
    id: str | None = None
    name: str | None = None
    input: JsonValue = None
    title: str | None = None


class ToolChoice(ExtraAllowModel):
    type: str | None = None
    name: str | None = None
    disable_parallel_tool_use: JsonValue = None


class OutputFormat(ExtraAllowModel):
    type: str | None = None
    schema_: JsonObject | None = Field(default=None, alias="schema")


class Message(ExtraAllowModel):
    role: str
    content: str | list[ContentBlock]


class Tool(ExtraAllowModel):
    name: str | None = None
    description: str | None = None
    input_schema: JsonObject | None = None
    type: str | None = None
    cache_control: CacheControl | None = None


class ThinkingConfig(ExtraAllowModel):
    type: str | None = None
    budget_tokens: int | None = None

    @field_validator("budget_tokens", mode="before")
    @classmethod
    def _drop_bool(cls, value: object) -> object:
        return None if isinstance(value, bool) else value


class OutputConfig(ExtraAllowModel):
    effort: str | None = None
    format: OutputFormat | None = None


class Metadata(ExtraAllowModel):
    user_id: str | None = None


class MessagesRequest(ExtraAllowModel):
    model: str | None = None
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[ContentBlock] | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    thinking: ThinkingConfig | None = None
    output_config: OutputConfig | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: Metadata | None = None
    service_tier: str | None = None
    stream: bool | None = None

    mcp_servers: list[JsonValue] | None = None
    container: JsonValue = None
    inference_geo: JsonValue = None
    context_management: JsonValue = None
    cache_control: CacheControl | None = None
