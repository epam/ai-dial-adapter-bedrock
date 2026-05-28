import json
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.upstream_config import ApiKeyUpstreamConfig

_UPSTREAM_KEY_HEADER = "x-upstream-key"
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
    api_key = request.headers.get(_UPSTREAM_KEY_HEADER)
    if not api_key:
        return Response(
            content=json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": f"Missing {_UPSTREAM_KEY_HEADER!r} header",
                    },
                }
            ),
            status_code=401,
            media_type="application/json",
        )

    body = await request.body()
    sdk_client = await create_anthropic_client(
        ApiKeyUpstreamConfig(api_key=api_key)
    )

    # sdk_client.default_headers includes x-api-key and anthropic-version.
    # Filter out Omit sentinels (headers the SDK wants omitted).
    headers: dict[str, str] = {
        k: v
        for k, v in sdk_client.default_headers.items()
        if isinstance(v, str)
    }
    headers["content-type"] = "application/json"
    for name in _FORWARDED_REQUEST_HEADERS:
        if value := request.headers.get(name):
            headers[name] = value

    http_req = sdk_client._client.build_request(  # type: ignore[union-attr]
        method=request.method,
        url=path,
        headers=headers,
        content=body or None,
        params=dict(request.query_params),
    )
    upstream = await sdk_client._client.send(  # type: ignore[union-attr]
        http_req, stream=True
    )

    if not upstream.is_success or not _is_streaming(body):
        content = await upstream.aread()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=_pick_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    async def _chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        content=_chunks(),
        status_code=upstream.status_code,
        headers=_pick_response_headers(upstream.headers),
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
