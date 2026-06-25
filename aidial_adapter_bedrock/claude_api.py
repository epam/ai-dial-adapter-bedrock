import contextlib
import json
from collections.abc import AsyncIterator
from logging import DEBUG

import httpx
from anthropic import AsyncAnthropicBedrock
from anthropic._models import FinalRequestOptions
from anthropic._streaming import ServerSentEvent, SSEDecoder
from anthropic.lib.bedrock._stream_decoder import AWSEventStreamDecoder
from botocore.eventstream import EventStreamBuffer
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.server.exceptions import (
    anthropic_exception_decorator,
)
from aidial_adapter_bedrock.upstream_config import parse_upstream_config
from aidial_adapter_bedrock.utils.json import json_dumps_short
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

app = FastAPI()


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


def _format_for_log(content: bytes) -> object:
    text = content.decode("utf-8", errors="replace")
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(text)
    return text


async def _aiter_and_close(
    source: AsyncIterator[bytes], response: httpx.Response
) -> AsyncIterator[bytes]:
    try:
        async for chunk in source:
            yield chunk
    finally:
        await response.aclose()


async def _log_sse_events(
    events: AsyncIterator[ServerSentEvent],
) -> AsyncIterator[ServerSentEvent]:
    async for sse in events:
        if log.isEnabledFor(DEBUG):
            data: object = sse.data
            with contextlib.suppress(json.JSONDecodeError):
                data = json.loads(sse.data)
            log.debug(f"response event: {json_dumps_short(data)}")
        yield sse


@anthropic_exception_decorator
async def _proxy(request: Request, path: str) -> Response:
    json_body = None
    if content := await request.body():
        with contextlib.suppress(json.JSONDecodeError):
            json_body = json.loads(content)

    is_streaming = _is_streaming_request(json_body, path)

    upstream_config = await parse_upstream_config(request)
    client = await create_anthropic_client(upstream_config)

    options = FinalRequestOptions.construct(
        method=request.method.lower(),
        url=path,
        json_data=json_body,
    )

    if log.isEnabledFor(DEBUG):
        log.debug(
            "request: "
            + json_dumps_short(
                {
                    "method": request.method,
                    "path": path,
                    "streaming": is_streaming,
                    "body": json_body,
                }
            )
        )

    response = await client.request(
        cast_to=httpx.Response,
        options=options,
        stream=is_streaming,
        stream_cls=None,
    )

    if is_streaming:
        out_headers = httpx.Headers(response.headers)
        out_headers.pop("Content-Encoding", None)
        out_headers.pop("Content-Length", None)
        out_headers["Content-Type"] = "text/event-stream"

        if isinstance(client, AsyncAnthropicBedrock):
            events = _bedrock_stream_to_sse(
                _aiter_and_close(response.aiter_raw(), response)
            )
        else:
            events = SSEDecoder().aiter_bytes(
                _aiter_and_close(response.aiter_bytes(), response)
            )

        stream = _sse_to_bytes_iterator(_log_sse_events(events))

        return StreamingResponse(
            content=stream,
            status_code=response.status_code,
            headers=out_headers,
        )

    else:
        response.headers.pop("Content-Encoding", None)
        response.headers.pop("Content-Length", None)

        content = await response.aread()
        if log.isEnabledFor(DEBUG):
            log.debug(
                "response: "
                + json_dumps_short(
                    {
                        "status": response.status_code,
                        "body": _format_for_log(content),
                    }
                )
            )

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
