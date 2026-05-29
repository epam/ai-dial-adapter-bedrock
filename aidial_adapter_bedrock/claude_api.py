from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.upstream_config import parse_upstream_config

app = FastAPI()


async def _proxy(request: Request, path: str) -> Response | StreamingResponse:
    body = await request.body()
    upstream_config = await parse_upstream_config(request)
    client = await create_anthropic_client(upstream_config)

    http_req = client._client.build_request(
        method=request.method,
        url=path,
        headers=request.headers,
        content=body or None,
        params=request.query_params,
    )
    response = await client._client.send(http_req, stream=True)

    if "text/event-stream" in response.headers.get("content-type"):

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
