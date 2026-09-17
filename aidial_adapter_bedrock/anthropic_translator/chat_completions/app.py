from collections.abc import Awaitable, Callable

from aidial_sdk.chat_completion.request import StreamOptions
from anthropic.types import Message
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from openai import AsyncOpenAI, AsyncStream, Omit
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    from_chat_completions,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.stop_emulation import (
    emulated_stop_sequences,
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
from aidial_adapter_bedrock.anthropic_translator.tool_names import (
    ToolNameAliases,
)

app: FastAPI = FastAPI()


async def _handle_messages(request: Request) -> Response:
    base_url: str = require_base_url()
    req: MessagesRequest = await parse_request(request)

    deployment: str = resolve_deployment(request.headers, req.model)
    stop_sequences: list[str] = emulated_stop_sequences(req, deployment)
    aliases: ToolNameAliases = ToolNameAliases()

    body: CoreChatCompletionRequest = to_chat_completions_request(
        req, deployment, aliases
    )
    client: AsyncOpenAI = core_chat_completions_client(base_url, deployment)
    headers: dict[str, str | Omit] = core_headers(request)

    body.stream = bool(req.stream)
    if body.stream:
        body.stream_options = StreamOptions(include_usage=True)
        events: AsyncStream[ChatCompletionChunk] = await client.post(
            "/chat/completions",
            cast_to=ChatCompletion,
            body=body.model_dump(mode="json", exclude_none=True),
            options={"headers": headers},
            stream=True,
            stream_cls=AsyncStream[ChatCompletionChunk],
        )
        return stream_response(
            translate_stream(events, deployment, aliases, stop_sequences)
        )

    response: ChatCompletion = await client.post(
        "/chat/completions",
        cast_to=ChatCompletion,
        body=body.model_dump(mode="json", exclude_none=True),
        options={"headers": headers},
    )
    translated: Message = from_chat_completions(
        response, deployment, aliases, stop_sequences
    )
    return JSONResponse(content=translated.model_dump(mode="json"))


_messages: Callable[[Request], Awaitable[Response]] = build_endpoint(
    _handle_messages
)

app.add_api_route("/v1/messages", _messages, methods=["POST"])


app.add_api_route(
    "/{full_path:path}",
    not_found,
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
