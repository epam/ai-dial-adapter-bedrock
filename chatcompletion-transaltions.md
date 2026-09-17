# Anthropic Messages to Chat Completions

Use `POST /to-chat-completions/anthropic/v1/messages` to send Anthropic Messages requests to a Chat Completions deployment through DIAL Core. Both JSON and streaming responses use the Anthropic Messages format.

## Contents

1. [Setup and routing](#setup-and-routing)
2. [Request options](#request-options)
3. [Conversation content](#conversation-content)
4. [Tools](#tools)
5. [Prompt caching](#prompt-caching)
6. [Responses and usage](#responses-and-usage)
7. [Errors and logging](#errors-and-logging)

## Setup and routing

See the [DIAL Core configuration example](README.md#anthropic-messages-to-chat-completions-translator) and [environment settings](README.md#environment-variables).

| Setting | Default | Purpose |
|---|---|---|
| `DIAL_URL` | Unset | Required DIAL Core base URL |
| `DIAL_API_VERSION` | `2025-01-01-preview` | Version sent to Core's Chat Completions endpoint |
| `LOG_LEVEL` | `INFO` | Set `DEBUG` to log request and response bodies |

Core's `x-dial-deployment-id` header selects the deployment; the request's `model` is the fallback. Core receives the translated request at `/openai/deployments/{deployment}/chat/completions`.

Authentication uses Core's per-request `api-key`. Distributed tracing and `x-dial-cache-policy` pass through. Inbound `authorization`, `x-api-key`, provider credentials, Anthropic protocol headers, and deployment-routing headers are excluded from the outgoing hop. Cache policy is validated by Core.

Only the Messages creation endpoint is supported. Unknown paths return an Anthropic-shaped 404.

<details>
<summary>Request mapping: Anthropic Messages → Chat Completions</summary>

| Anthropic | Chat Completions |
|---|---|
| `model` | `model`; `x-dial-deployment-id` takes precedence |
| `system`, system messages, `mid_conv_system` | One leading `system` message |
| User/assistant text | User content parts / assistant `content` |
| Images and documents | Image URL or file/text content parts |
| `tool_use` | Assistant `tool_calls` |
| `tool_result` | `tool` message; residual images become user content |
| `tools`, `tool_choice` | Function tools, `tool_choice`, `parallel_tool_calls` |
| `max_tokens` | `max_completion_tokens` |
| `temperature`, `top_p` | Same fields |
| Thinking and `output_config.effort` | `reasoning_effort` |
| `output_config.format` | Strict JSON-schema `response_format` |
| `metadata.user_id` | `user` |
| `service_tier` | `auto` / `default` |
| `stop_sequences` | `stop` for every deployment |
| `stream` | `stream` and `stream_options.include_usage` |
| `cache_control` | DIAL cache-breakpoint `custom_fields` |

</details>

<details>
<summary>Response mapping: Chat Completions → Anthropic Messages</summary>

| Chat Completions | Anthropic Messages |
|---|---|
| `id`, `model` | `id`, `model` |
| `message.content`, `message.refusal` | Text content blocks |
| `message.tool_calls` | `tool_use` blocks |
| Native thinking / reasoning stages | Thinking blocks |
| URL citations | Synthesized web-search tool-use/result blocks |
| `finish_reason` and response content | `stop_reason`; `stop_sequence` is `null` |
| `prompt_tokens` | `input_tokens`, minus cache reads/writes |
| `completion_tokens` | `output_tokens` |
| Cached/cache-write tokens | Cache input-token counters |
| Reasoning tokens | `output_tokens_details.thinking_tokens` |
| Streaming deltas | Anthropic content-block SSE events |
| Upstream errors | Anthropic error envelope or terminal SSE error |

</details>

## Request options

DIAL Core validates requests before forwarding them to the translator. The translator reads the JSON using Anthropic SDK request types for static typing, maps it to Chat Completions, sends it to Core, and translates the response back. It does not perform request-schema validation. Unused fields are ignored.

| Anthropic option | Chat Completions behavior |
|---|---|
| `max_tokens` | Forwarded as `max_completion_tokens`, without clamping |
| `temperature`, `top_p` | Forwarded when provided |
| `stop_sequences` | Forwarded as `stop` for every deployment, regardless of its name |
| `output_config.format` | A `json_schema` format with a nonempty schema requests strict structured output |
| `service_tier` | `auto` and `standard_only` become `auto` and `default` |
| `metadata.user_id` | Forwarded whole as `user` when nonempty |
| `stream` | Selects an Anthropic SSE response |

Options are sent without consulting `x-dial-deployment-features`. The target deployment can reject options it does not support. Token counting and output-cap clamping are outside the translator's scope.

### Reasoning

`thinking.type: disabled` emits `reasoning_effort: none`. Otherwise, explicit `output_config.effort` is forwarded using the Anthropic SDK's accepted values. The target deployment can reject an effort it does not support.

Without an explicit effort, the deployment default applies, including for adaptive or enabled thinking. `thinking.budget_tokens` is not forwarded: a token budget has no protocol-defined conversion to a qualitative reasoning effort.

### Stop sequences

Stop sequences are forwarded to the target as `stop`. The translator does not infer capabilities from deployment names or emulate stops locally. Unsupported parameters are handled by the target deployment.

Chat Completions does not identify which sequence matched, so responses use `stop_sequence: null`; a normal upstream `stop` finish reason maps to `end_turn`.

## Conversation content

All system instructions become one leading system message, joined with blank lines. This includes top-level `system`, system-role turns, and `mid_conv_system` blocks on any role, in conversation order.

| Supported content | Translation |
|---|---|
| User and assistant strings | One message when nonempty |
| User text blocks | Text content parts |
| User base64 and URL images | Image URL parts; base64 images become data URLs |
| User base64 documents | Inline file parts; defaults are `document.pdf` and `application/pdf` |
| User text documents | Plain text |
| User tool results | Tool messages, followed by any residual user content |
| Assistant text blocks | Joined with newlines |
| Assistant tool use | Function tool calls |

For tool results, text is joined with newlines, images move into residual user content, and `is_error: true` prefixes text with `Error: `. Other nested content is ignored. Empty strings produce no messages.

System relocation and grouping tool results before user content can change the original interleaving. Assistant thinking blocks are omitted from replayed history. SDK-defined content types without a supported translation are dropped with a warning.

A top-level document block with `citations.enabled` requests citation output from the deployment.

## Tools

Named tools with no `type` or `type: custom` become function tools. Descriptions and input schemas are forwarded; the root `$schema` key is removed. Custom tools require a name and input schema as defined by the SDK; translated tool definitions use `strict: false`.

| Tool choice | Outgoing behavior |
|---|---|
| `auto` | Automatic selection |
| `any` | At least one tool required |
| `none` | No tool use |
| `tool` | The named function |
| `disable_parallel_tool_use` | Inverted into `parallel_tool_calls`, independently of the choice type |

Tool names outside `^[a-zA-Z_][a-zA-Z0-9_-]{2,63}$` are aliased for the target deployment and restored in the response. Restoration has a process-local limit of 4096 aliases; eviction can prevent restoring an older name.

Server tools are outside the supported set. Web-search results can still appear in responses when the deployment returns URL citations.

## Prompt caching

Anthropic cache markers become DIAL `custom_fields.cache_breakpoint` markers:

| Marker source | Placement |
|---|---|
| System content | The merged system message |
| A user or assistant content block | Every message produced from that turn |
| A tool definition | The converted function tool |
| Top-level request `cache_control` | The last converted user message |

Cache markers require `type: ephemeral`, including the top-level shorthand. A final tool-only turn or assistant prefill is skipped when locating the last user message. Nested markers inside tool results are not inspected.

The SDK-supported TTLs, `5m` and `1h`, become absolute UTC expiry timestamps. When markers merge, the later expiry wins. Unreadable TTLs leave the default expiry. Without a TTL, Core/provider defaults apply.

Markers cover message prefixes and are coarser than Anthropic block markers. They are sent without a breakpoint-count limit; usefulness depends on the target deployment's support for DIAL cache markers.

## Responses and usage

Only the first completion choice is translated. Content appears in this order: thinking, synthesized web-search results, text, refusal text, and function tool calls. Empty responses contain an empty text block. Malformed or non-object tool arguments become `{}`.

Non-streaming thinking uses the first nonempty native thinking block and its signature, falling back to reasoning-stage text. Streaming uses reasoning-stage text and a signature when it arrives while the thinking block remains open. A late signature is dropped with a warning.

URL citations become a synthesized web-search call/result pair. Non-streaming groups all results in one pair. Streaming groups newly seen URLs per chunk and deduplicates them across chunks.

Stop reasons follow this precedence: function tool use, output length limit, refusal/content filter, then `end_turn`.

Streaming sends `message_start`, `ping`, content-block events, `message_delta`, and `message_stop`. Interleaved parallel tool calls can leave multiple tool blocks open. A terminal error can interrupt an open block and ends with an `error` event.

| Upstream usage | Anthropic usage |
|---|---|
| Prompt tokens minus cache reads and writes, floored at zero | `input_tokens` |
| Completion tokens | `output_tokens` |
| Cached prompt tokens | `cache_read_input_tokens` |
| Cache-write prompt tokens (`cache_write_tokens` or `cacheWriteTokens`) | `cache_creation_input_tokens` |
| Nonzero reasoning tokens | `output_tokens_details.thinking_tokens` |

Missing/null counters become zero. Thinking tokens remain included in the output total. Cache-write usage depends on the deployment reporting it. Streaming usage is included in the terminal `message_delta`.

## Errors and logging

Errors have the Anthropic envelope `{"type":"error","error":{"type":"...","message":"..."}}`.

Validation errors returned by Core are translated like other upstream errors. Upstream error statuses are preserved: 400/422 map to invalid request, 401 to authentication, 403 to permission, 404 to not found, 413 to request too large, 429 to rate limit, and 503/529 to overloaded. Other upstream errors use `api_error`.

Connection failures return 502. Missing server configuration and unexpected internal failures return 500 with a generic message; diagnostic details go to server logs. Before streaming starts, errors return JSON with an HTTP status. After streaming starts, errors use a terminal SSE error event.

The project’s `bedrock` logger records conversion warnings and errors. `LOG_LEVEL=DEBUG` includes inbound request and outgoing response bodies or individual SSE chunks.
