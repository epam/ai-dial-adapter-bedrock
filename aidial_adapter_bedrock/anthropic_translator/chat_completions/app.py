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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from openai import AsyncOpenAI, AsyncStream, Omit
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    DeploymentProfile,
    get_deployment_profile,
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
    prepare,
    stream_response,
)
from aidial_adapter_bedrock.anthropic_translator.core_client import (
    caller_credential,
    core_chat_completions_client,
    core_headers,
)
from aidial_adapter_bedrock.anthropic_translator.stop_sequences import (
    strips_stop_parameter,
)

app = FastAPI()


async def _handle_messages(request: Request) -> Response:
    # `model` is also the DIAL deployment name here: Chat Completions addresses
    # models per-deployment, unlike the Responses API.
    base_url, req, deployment = await prepare(request)
    profile: DeploymentProfile = await get_deployment_profile(
        base_url, caller_credential(request.headers), deployment
    )

    body: CoreChatCompletionRequest
    body, aliases = to_chat_completions_request(req, deployment, profile)
    data = body.model_dump(mode="json", exclude_none=True, exclude={"stream"})
    # `custom_fields` isn't part of the SDK's typed `create()` signature, so
    # it rides in `extra_body` instead (only set when non-empty).
    custom_fields = data.pop("custom_fields", None)
    extra_body = {"custom_fields": custom_fields} if custom_fields else None
    client: AsyncOpenAI = core_chat_completions_client(base_url, deployment)
    headers: dict[str, str | Omit] = core_headers(request.headers)

    # Sequences the outbound body deliberately omitted, to be reproduced here.
    emulated_stop: list[str] = (
        req.stop_sequences or [] if strips_stop_parameter(deployment) else []
    )

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
