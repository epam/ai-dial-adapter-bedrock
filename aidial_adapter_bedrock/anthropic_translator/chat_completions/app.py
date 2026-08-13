"""
FastAPI sub-app for the Anthropic Messages → OpenAI Chat Completions
translator.

Mounted at `/to-chat-completions/anthropic` in the main app, exposing:

- `POST /v1/messages`

There is no `/v1/messages/count_tokens` route here: Chat Completions has no
token-counting endpoint to forward to, so an unknown path (including that
one) falls through to the shared 404.

A pure HTTP protocol translator: it re-enters DIAL Core at
`/openai/deployments/{deployment}/chat/completions` and never touches AWS.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from openai import AsyncOpenAI, AsyncStream, Omit
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    DeploymentProfile,
    parse_deployment_profile,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    from_chat_completions,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.streaming import (
    translate_stream,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.to_chat_completions import (
    CoreChatCompletionRequest,
    to_chat_completions_request,
)
from aidial_adapter_bedrock.anthropic_translator.common import (
    build_endpoint,
    not_found,
    parse_request,
    require_base_url,
    resolve_deployment,
    stream_response,
)
from aidial_adapter_bedrock.anthropic_translator.core_client import (
    core_chat_completions_client,
    core_headers,
)
from aidial_adapter_bedrock.anthropic_translator.stop_sequences import (
    emulated_stop_sequences,
)
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)

app = FastAPI()


async def _handle_messages(request: Request) -> Response:
    base_url: str = require_base_url()
    req: MessagesRequest = await parse_request(request)
    # Chat Completions addresses models per deployment, so this names both the
    # URL path segment and the body's `model`.
    deployment: str = resolve_deployment(request, req)
    # Capabilities arrive on the inbound request, so this handler makes exactly
    # one outbound call.
    profile: DeploymentProfile = parse_deployment_profile(request.headers)
    aliases: ToolNameAliases = ToolNameAliases()
    emulated_stop: list[str] = emulated_stop_sequences(req, deployment)

    body: CoreChatCompletionRequest = to_chat_completions_request(
        req, deployment, profile, aliases
    )
    data: dict[str, Any] = body.model_dump(
        mode="json", exclude_none=True, exclude={"stream"}
    )
    # `custom_fields` isn't part of the SDK's typed `create()` signature, so
    # it rides in `extra_body` instead.
    custom_fields = data.pop("custom_fields", None)
    extra_body = {"custom_fields": custom_fields} if custom_fields else None

    client: AsyncOpenAI = core_chat_completions_client(base_url, deployment)
    headers: dict[str, str | Omit] = core_headers(request.headers)

    if req.stream:
        events: AsyncStream[
            ChatCompletionChunk
        ] = await client.chat.completions.create(
            **data,
            stream=True,
            # Without this the final usage is absent and the client's cost and
            # cache telemetry read zero.
            stream_options={"include_usage": True},
            extra_headers=headers,
            extra_body=extra_body,
        )
        return stream_response(
            translate_stream(events, deployment, aliases, emulated_stop)
        )

    response: ChatCompletion = await client.chat.completions.create(
        **data, stream=False, extra_headers=headers, extra_body=extra_body
    )
    return JSONResponse(
        content=from_chat_completions(
            response, deployment, aliases, emulated_stop
        ).model_dump(mode="json")
    )


_messages = build_endpoint(_handle_messages)

app.add_api_route("/v1/messages", _messages, methods=["POST"])

# Catch-all: unknown paths (including /v1/messages/count_tokens) → 404 in Anthropic shape.
app.add_api_route(
    "/{full_path:path}",
    not_found,
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
