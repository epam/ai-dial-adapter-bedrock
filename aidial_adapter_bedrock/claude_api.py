import contextlib
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from functools import wraps

import httpx
from anthropic import AsyncAnthropicBedrock
from anthropic._models import FinalRequestOptions
from anthropic._streaming import ServerSentEvent
from anthropic.lib.bedrock._stream_decoder import AWSEventStreamDecoder
from botocore.eventstream import EventStreamBuffer
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import Headers as StarletteHeaders

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.server.exceptions import (
    anthropic_exception_decorator,
    dial_exception_decorator,
)
from aidial_adapter_bedrock.upstream_config import parse_upstream_config
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

app = FastAPI()

_Content = str | bytes | memoryview
_Handler = Callable[[Request, str], Awaitable[Response]]


def _is_streaming_request(body: dict | None, path: str) -> bool:
    # Note that it isn't enough to check the response header
    # content-type to be equal to text/event-stream, since Bedrock
    # returns the stream in its own event stream format:
    # content-type:application/vnd.amazon.eventstream
    if path == "/v1/messages" and isinstance(body, dict):
        return bool(body.get("stream"))

    return False


async def _bedrock_stream_to_sse(
    iterator: AsyncIterator[bytes],
) -> AsyncIterator[ServerSentEvent]:
    decoder = AWSEventStreamDecoder()
    event_stream_buffer = EventStreamBuffer()
    async for chunk in iterator:
        event_stream_buffer.add_data(chunk)
        for event in event_stream_buffer:
            message = decoder._parse_message_from_event(event)
            if message:
                event = "completion"
                with contextlib.suppress(Exception):
                    event = json.loads(message).get("type")
                yield ServerSentEvent(data=message, event=event)


async def _sse_to_bytes_iterator(
    event: AsyncIterator[ServerSentEvent],
) -> AsyncIterator[bytes]:
    async for sse in event:
        if sse.id is not None:
            yield f"id: {sse.id}\n".encode()
        if sse.event is not None:
            yield f"event: {sse.event}\n".encode()
        for line in sse.data.split("\n"):
            yield f"data: {line}\n".encode()
        if sse.retry is not None:
            yield f"retry: {sse.retry}\n".encode()
        yield b"\n"


def _strip_content_headers(response_headers: httpx.Headers) -> None:
    # The adapter decodes the response body before forwarding it, so the
    # Content-Encoding header no longer applies. Leaving it would cause the
    # downstream client to attempt a second decompression and fail.
    response_headers.pop("Content-Encoding", None)
    # Content-Length reflected the compressed size; after decoding it no longer
    # matches the body, so drop it and let the framework recompute it.
    # And even when the content was uncompressed to begin with,
    # the content length can change do to the SSE reformatting.
    response_headers.pop("Content-Length", None)


def _as_text(data: _Content) -> str:
    if isinstance(data, str):
        return data
    return bytes(data).decode("utf-8", errors="replace")


async def _log_stream_chunks(
    iterator: AsyncIterable[_Content],
) -> AsyncIterator[_Content]:
    async for chunk in iterator:
        with contextlib.suppress(Exception):
            log.debug(f"response chunk: {_as_text(chunk)}")
        yield chunk


def _logging_decorator(func: _Handler) -> _Handler:
    @wraps(func)
    async def wrapper(request: Request, path: str) -> Response:
        if not log.isEnabledFor(logging.DEBUG):
            return await func(request, path)

        with contextlib.suppress(Exception):
            log.debug(f"request: {_as_text(await request.body())}")

        response = await func(request, path)

        if isinstance(response, StreamingResponse):
            response.body_iterator = _log_stream_chunks(response.body_iterator)
        else:
            with contextlib.suppress(Exception):
                log.debug(f"response: {_as_text(response.body)}")

        return response

    return wrapper


_UNSUPPORTED_BEDROCK_ANTHROPIC_BETA_FLAGS = {
    "oauth-2025-04-20",
    "redact-thinking-2026-02-12",
    "thinking-token-count-2026-05-13",
    "prompt-caching-scope-2026-01-05",
    "claude-code-20250219",
    "advisor-tool-2026-03-01",
}


def _adapt_anthropic_beta_for_bedrock(value: str | None) -> str | None:
    if value is None:
        return None
    features = [
        feature
        for feature in value.split(",")
        if feature not in _UNSUPPORTED_BEDROCK_ANTHROPIC_BETA_FLAGS
    ]
    return ",".join(features) or None


def _on_dict_value(
    dct: dict[str, str], kye: str, func: Callable[[str | None], str | None]
) -> dict[str, str]:
    value = func(dct.get(kye))
    if value is None:
        dct.pop(kye, None)
    else:
        dct[kye] = value
    return dct


def _build_request_headers(
    headers: StarletteHeaders, *, is_bedrock: bool
) -> dict[str, str]:
    def _keep_header(header: str) -> bool:
        header = header.lower()
        return header.startswith("anthropic-") or header == "accept-encoding"

    result = {k.lower(): v for (k, v) in headers.items() if _keep_header(k)}

    if is_bedrock:
        result = _on_dict_value(
            result, "anthropic-beta", _adapt_anthropic_beta_for_bedrock
        )

    return result


@dial_exception_decorator
@anthropic_exception_decorator
@_logging_decorator
async def _proxy(request: Request, path: str) -> Response:
    json_body = None
    if content := await request.body():
        with contextlib.suppress(json.JSONDecodeError):
            json_body = json.loads(content)

    is_streaming = _is_streaming_request(json_body, path)

    upstream_config = await parse_upstream_config(request)
    client = await create_anthropic_client(upstream_config)

    headers = _build_request_headers(
        request.headers, is_bedrock=isinstance(client, AsyncAnthropicBedrock)
    )
    if log.isEnabledFor(logging.DEBUG):
        # Ask the upstream not to compress the response so its body (and
        # streamed chunks) can be logged as-is. Forgoing compression on the
        # upstream hop is a fair price for painless logging.
        headers["accept-encoding"] = "identity"

    options = FinalRequestOptions.construct(
        method=request.method.lower(),
        url=path,
        json_data=json_body,
        headers=headers,
    )

    response = await client.request(
        cast_to=httpx.Response,
        options=options,
        stream=is_streaming,
        stream_cls=None,
    )

    if is_streaming:

        async def _stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        stream = _stream()
        if isinstance(client, AsyncAnthropicBedrock):
            response.headers["Content-Type"] = "text/event-stream"
            _strip_content_headers(response.headers)
            stream = _sse_to_bytes_iterator(_bedrock_stream_to_sse(stream))

        return StreamingResponse(
            content=stream,
            status_code=response.status_code,
            headers=response.headers,
        )

    else:
        content = await response.aread()
        _strip_content_headers(response.headers)
        return Response(
            content=content,
            status_code=response.status_code,
            headers=response.headers,
        )


def _create_proxy_handler(path: str):
    async def handler(request: Request) -> Response:
        return await _proxy(request, path)

    return handler


_PROXIED_ENDPOINTS = [
    ("POST", "/v1/messages"),
    ("POST", "/v1/messages/batches"),
    ("POST", "/v1/messages/count_tokens"),
    ("GET", "/v1/models"),
]

for method, path in _PROXIED_ENDPOINTS:
    app.router.add_api_route(
        path=path, methods=[method], endpoint=_create_proxy_handler(path)
    )
