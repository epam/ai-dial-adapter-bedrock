"""
The re-entrant call into DIAL Core.

Having converted an Anthropic request into a Chat Completions request, the
translator sends it back to DIAL Core — at
`{DIAL_URL}/openai/deployments/{deployment}/chat/completions` — using the
`openai` SDK.

The SDK client itself is stateless and created per request; all of them share a
single `httpx.AsyncClient` connection pool, created lazily and closed in the
app lifespan. The caller's credential and tracing headers travel per request
via `core_headers`.
"""

from functools import cache

import httpx
from fastapi import Request
from openai import AsyncOpenAI, Omit, omit

_CONNECT_TIMEOUT = 5.0

# Headers forwarded to Core so the inner leg joins the same distributed trace.
_TRACE_HEADERS = (
    "traceparent",
    "tracestate",
    "b3",
    "x-b3-traceid",
    "x-b3-spanid",
    "x-b3-parentspanid",
    "x-b3-sampled",
)

# The SDK insists on an api_key; the real credential is a per-request header.
_PLACEHOLDER_API_KEY = "-"

# The `api-version` query param some Chat Completions deployments (e.g.Azure-backed) require.
_API_VERSION = "2025-01-01-preview"


@cache
def _get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        # No read timeout: a streaming response may idle between chunks.
        timeout=httpx.Timeout(
            _CONNECT_TIMEOUT, read=None, write=None, pool=_CONNECT_TIMEOUT
        ),
    )


async def close_http_client() -> None:
    # Nothing to close until the first request has built the client.
    if _get_http_client.cache_info().currsize:
        await _get_http_client().aclose()
        _get_http_client.cache_clear()


def core_chat_completions_client(base_url: str, deployment: str) -> AsyncOpenAI:
    """An OpenAI client aimed at DIAL Core's Chat Completions API for aspecific deployment."""
    return AsyncOpenAI(
        api_key=_PLACEHOLDER_API_KEY,
        base_url=f"{base_url}/openai/deployments/{deployment}",
        max_retries=0,
        default_query={"api-version": _API_VERSION},
        http_client=_get_http_client(),
    )


def core_headers(request: Request) -> dict[str, str | Omit]:
    """Headers for the call to Core: the caller's credential in the same header
    it arrived in, plus any tracing headers.

    Anthropic-specific headers (`anthropic-version`, `anthropic-beta`) are
    deliberately NOT forwarded — neither upstream understands them and they
    must not leak.
    """
    # `omit` suppresses the bearer header the SDK would derive from the
    # placeholder api_key. A real bearer token overwrites it below.
    headers: dict[str, str | Omit] = {"Authorization": omit}

    if api_key := request.headers.get("api-key"):
        headers["api-key"] = api_key
    elif authorization := request.headers.get("authorization"):
        headers["Authorization"] = authorization

    for name in _TRACE_HEADERS:
        if value := request.headers.get(name):
            headers[name] = value

    return headers
