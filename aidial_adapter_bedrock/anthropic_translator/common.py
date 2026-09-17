import contextlib
import logging
import os
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from functools import wraps

from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_bedrock.anthropic_translator.errors import (
    INTERNAL_ERROR_MESSAGE,
    AnthropicErrorType,
    AnthropicHTTPError,
    anthropic_error_response,
    translator_error_handler,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

NOT_CONFIGURED: str = "translator is not configured (DIAL_URL is not set)"


def require_base_url() -> str:
    url: str | None = os.getenv("DIAL_URL")
    if not url:
        log.error(NOT_CONFIGURED)
        raise AnthropicHTTPError(AnthropicErrorType.API, INTERNAL_ERROR_MESSAGE)

    return url.rstrip("/")


def stream_response(chunks: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(
        chunks,
        media_type="text/event-stream",
        headers={"cache-control": "no-cache"},
    )


async def not_found(request: Request) -> Response:
    return anthropic_error_response(
        AnthropicErrorType.NOT_FOUND, f"Unknown endpoint: {request.url.path}"
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
    return with_debug_logging(translator_error_handler(handler))
