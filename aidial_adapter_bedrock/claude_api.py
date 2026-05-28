import json
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.upstream_config import parse_upstream_config

_FORWARDED_REQUEST_HEADERS: frozenset[str] = frozenset(
    {"anthropic-version", "anthropic-beta"}
)
_FORWARDED_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {"request-id", "retry-after", "x-request-id"}
)

app = FastAPI()


def _is_streaming(body: bytes) -> bool:
    if not body:
        return False
    try:
        return bool(json.loads(body).get("stream", False))
    except Exception:
        return False


def _pick_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {k: headers[k] for k in _FORWARDED_RESPONSE_HEADERS if k in headers}


async def _proxy(request: Request, path: str) -> Response | StreamingResponse:
    body = await request.body()
    upstream_config = await parse_upstream_config(request)
    client = await create_anthropic_client(upstream_config)

    # sdk_client.default_headers includes x-api-key and anthropic-version.
    # Filter out Omit sentinels (headers the SDK wants omitted).
    headers: dict[str, str] = {
        k: v for k, v in client.default_headers.items() if isinstance(v, str)
    }
    headers["content-type"] = "application/json"
    for name in _FORWARDED_REQUEST_HEADERS:
        if value := request.headers.get(name):
            headers[name] = value

    http_req = client._client.build_request(
        method=request.method,
        url=path,
        headers=headers,
        content=body or None,
        params=dict(request.query_params),
    )
    response = await client._client.send(http_req, stream=True)

    if not response.is_success or not _is_streaming(body):
        content = await response.aread()
        return Response(
            content=content,
            status_code=response.status_code,
            headers=_pick_response_headers(response.headers),
            media_type=response.headers.get("content-type", "application/json"),
        )

    async def _chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()

    return StreamingResponse(
        content=_chunks(),
        status_code=response.status_code,
        headers=_pick_response_headers(response.headers),
        media_type="text/event-stream",
    )


@app.post("/v1/messages", response_model=None)
async def messages(request: Request) -> Response | StreamingResponse:
    return await _proxy(request, "/v1/messages")


@app.post("/v1/messages/batches", response_model=None)
async def message_batches(request: Request) -> Response | StreamingResponse:
    return await _proxy(request, "/v1/messages/batches")


@app.post("/v1/messages/count_tokens", response_model=None)
async def count_tokens(request: Request) -> Response | StreamingResponse:
    return await _proxy(request, "/v1/messages/count_tokens")


@app.get("/v1/models", response_model=None)
async def models(request: Request) -> Response | StreamingResponse:
    return await _proxy(request, "/v1/models")
