import contextlib
import json
from collections.abc import AsyncIterator

import httpx
from anthropic._models import FinalRequestOptions
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.server.exceptions import (
    anthropic_exception_decorator,
)
from aidial_adapter_bedrock.upstream_config import parse_upstream_config

app = FastAPI()


def _is_streaming_request(body: dict | None, path: str) -> bool:
    # Note that it isn't enough to check the response header
    # content-type to be equal to text/event-stream, since Bedrock
    # returns the stream in its own event stream format:
    # content-type:application/vnd.amazon.eventstream
    if path != "/v1/messages" and isinstance(body, dict):
        return bool(body.get("stream"))

    return False


@anthropic_exception_decorator
async def _proxy(request: Request, path: str) -> Response:
    json_body = None
    if content := await request.body():
        with contextlib.suppress(json.JSONDecodeError):
            json_body = json.loads(content)

    stream = _is_streaming_request(json_body, path)

    upstream_config = await parse_upstream_config(request)
    client = await create_anthropic_client(upstream_config)

    options = FinalRequestOptions.construct(
        method=request.method.lower(),
        url=path,
        json_data=json_body,
    )

    response = await client.request(
        cast_to=httpx.Response,
        options=options,
        stream=True,
        stream_cls=None,
    )

    if stream:

        async def _stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            content=_stream(),
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    else:
        return Response(
            content=await response.aread(),
            status_code=response.status_code,
            headers=dict(response.headers),
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
