"""
Request-handling helpers for the Anthropic Messages translators: parsing the
inbound body, resolving the deployment/model, the Anthropic-shaped 404, and a
debug-logging decorator.
"""

import contextlib
import json
import logging
import os
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from functools import wraps
from typing import Any

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from aidial_adapter_bedrock.anthropic_translator.anthropic_api import (
    MessagesRequest,
)
from aidial_adapter_bedrock.anthropic_translator.errors import (
    API_ERROR,
    INVALID_REQUEST_ERROR,
    AnthropicHTTPError,
    anthropic_error_response,
    format_validation_error,
    translator_error_handler,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

NOT_CONFIGURED = "translator is not configured (DIAL_URL is not set)"


async def parse_request(request: Request) -> MessagesRequest:
    """Parse the body by hand. FastAPI's pydantic body binding would produce a
    FastAPI-shaped 422, which violates the Anthropic-shaped-errors-always
    rule."""
    raw: bytes = await request.body()
    try:
        body: Any = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AnthropicHTTPError(
            400,
            INVALID_REQUEST_ERROR,
            f"Request body is not valid JSON: {e}",
        ) from e
    if not isinstance(body, dict):
        raise AnthropicHTTPError(
            400, INVALID_REQUEST_ERROR, "Request body must be a JSON object"
        )
    try:
        return MessagesRequest.model_validate(body)
    except ValidationError as e:
        raise AnthropicHTTPError(
            400, INVALID_REQUEST_ERROR, format_validation_error(e)
        ) from e


def resolve_deployment(request: Request, req: MessagesRequest) -> str:
    """The deployment to address, which `x-dial-deployment-id` names when
    present: Core sets it to what it actually routed to, and that can differ
    from the body's `model`."""
    deployment = request.headers.get("x-dial-deployment-id") or req.model
    if not deployment:
        raise AnthropicHTTPError(
            400, INVALID_REQUEST_ERROR, "'model' is required"
        )
    return deployment


def require_base_url() -> str:
    """DIAL Core's base URL, read at call time so tests and runtime
    reconfiguration see the current environment."""
    url: str | None = os.getenv("DIAL_URL")
    if not url:
        raise AnthropicHTTPError(500, API_ERROR, NOT_CONFIGURED)
    # Composes cleanly with the `/openai/...` suffixes the clients append.
    return url.rstrip("/")


def stream_response(chunks: AsyncIterator[bytes]) -> StreamingResponse:
    """The response shape every streaming translator endpoint returns."""
    return StreamingResponse(
        chunks,
        media_type="text/event-stream",
        headers={"cache-control": "no-cache"},
    )


async def not_found(request: Request) -> Response:
    return anthropic_error_response(
        404, "not_found_error", f"Unknown endpoint: {request.url.path}"
    )


def _as_text(data: str | bytes | memoryview) -> str:
    if isinstance(data, str):
        return data
    return bytes(data).decode("utf-8", errors="replace")


async def _log_stream_chunks(
    iterator: AsyncIterable[str | bytes | memoryview],
) -> AsyncIterator[str | bytes | memoryview]:
    async for chunk in iterator:
        with contextlib.suppress(Exception):
            log.debug(f"response chunk: {_as_text(chunk).rstrip()}")
        yield chunk


def with_debug_logging(
    func: Callable[[Request], Awaitable[Response]],
) -> Callable[[Request], Awaitable[Response]]:
    """Log the request and response bodies — or attach a chunk logger to a
    streaming response — when DEBUG logging is enabled.

    Every step is suppressed: a logging failure must never fail a request.
    """

    def one_line(text: str) -> str:
        return "".join(text.splitlines())

    @wraps(func)
    async def wrapper(request: Request) -> Response:
        if not log.isEnabledFor(logging.DEBUG):
            return await func(request)

        with contextlib.suppress(Exception):
            log.debug(f"request: {one_line(_as_text(await request.body()))}")

        response: Response = await func(request)

        if isinstance(response, StreamingResponse):
            response.body_iterator = _log_stream_chunks(response.body_iterator)
        else:
            with contextlib.suppress(Exception):
                log.debug(f"response: {one_line(_as_text(response.body))}")

        return response

    return wrapper


def build_endpoint(
    handler: Callable[[Request], Awaitable[Response]],
) -> Callable[[Request], Awaitable[Response]]:
    """Wrap a raw endpoint implementation in the standard translator middleware:
    Anthropic-shaped error mapping, then debug logging on the outside."""
    return with_debug_logging(translator_error_handler(handler))
