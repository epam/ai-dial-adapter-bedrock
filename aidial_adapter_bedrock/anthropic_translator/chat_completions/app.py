import json
import logging

from anthropic.types.beta.message_create_params import MessageCreateParams
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_bedrock.anthropic_translator.chat_completions.from_chat_completions import (
    to_anthropic_response,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.streaming import (
    translate_stream,
)
from aidial_adapter_bedrock.anthropic_translator.chat_completions.to_chat_completions import (
    to_chat_completions_request,
)
from aidial_adapter_bedrock.anthropic_translator.common import (
    build_endpoint,
    not_found,
)
from aidial_adapter_bedrock.anthropic_translator.core_client import (
    core_chat_completions_client,
    core_headers,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    AnthropicErrorType,
    AnthropicHTTPError,
)
from aidial_adapter_bedrock.anthropic_translator.settings import get_dial_url
from aidial_adapter_bedrock.anthropic_translator.translation_state import (
    TranslationState,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

app = FastAPI()


async def _handle_messages(request: Request) -> Response:
    dial_url = get_dial_url()
    translation_state = TranslationState()

    # The body isn't schema-validated, so a malformed one fails while it is
    # read; report that as the client's error rather than a 500.
    try:
        req: MessageCreateParams = await request.json()
        deployment = request.headers.get("x-dial-deployment-id") or req["model"]
        chat_request = to_chat_completions_request(
            req, deployment, translation_state
        )
    except (
        AssertionError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        log.warning(f"Rejecting a malformed request body: {e!r}", exc_info=True)
        raise AnthropicHTTPError(
            AnthropicErrorType.INVALID_REQUEST,
            f"Malformed request body: {type(e).__name__}: {e}",
        ) from e

    body = chat_request.model_dump(mode="json", exclude_none=True)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"Translated request to Core: {json.dumps(body)}")

    client = core_chat_completions_client(dial_url)
    headers = core_headers(request)

    if chat_request.stream:
        events = await client.post(
            "/chat/completions",
            cast_to=ChatCompletion,
            body=body,
            options={"headers": headers},
            stream=True,
            stream_cls=AsyncStream[ChatCompletionChunk],
        )
        return StreamingResponse(
            translate_stream(events, deployment, translation_state),
            media_type="text/event-stream",
        )

    response = await client.post(
        "/chat/completions",
        cast_to=ChatCompletion,
        body=body,
        options={"headers": headers},
    )
    message = to_anthropic_response(response, deployment, translation_state)
    return JSONResponse(
        content=message.model_dump(mode="json", exclude_unset=True)
    )


app.add_api_route(
    "/v1/messages", build_endpoint(_handle_messages), methods=["POST"]
)


app.add_api_route(
    "/{full_path:path}",
    not_found,
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
