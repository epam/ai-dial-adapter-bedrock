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
from openai import AsyncOpenAI, Omit, omit
from starlette.datastructures import Headers

from aidial_adapter_bedrock.anthropic_translator.settings import get_api_version

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


@cache
def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        # No read timeout: a streaming response may idle between chunks.
        # Callers that need one (the model catalog) pass it per request.
        timeout=httpx.Timeout(
            _CONNECT_TIMEOUT, read=None, write=None, pool=_CONNECT_TIMEOUT
        ),
    )


async def close_http_client() -> None:
    # Nothing to close until the first request has built the client.
    if get_http_client.cache_info().currsize:
        await get_http_client().aclose()
        get_http_client.cache_clear()


def core_chat_completions_client(base_url: str, deployment: str) -> AsyncOpenAI:
    """An OpenAI client aimed at DIAL Core's Chat Completions API for a specific deployment."""
    return AsyncOpenAI(
        api_key=_PLACEHOLDER_API_KEY,
        base_url=f"{base_url}/openai/deployments/{deployment}",
        max_retries=0,
        default_query={"api-version": get_api_version()},
        http_client=get_http_client(),
    )


def caller_credential(headers: Headers) -> tuple[str, str] | None:
    """The caller's credential as the header name and value to resend it under.

    The translator holds no credentials of its own; Core accepts either form,
    including a raw key presented as `Authorization: Bearer <key>`.
    """
    if api_key := headers.get("api-key"):
        return "api-key", api_key
    if authorization := headers.get("authorization"):
        return "Authorization", authorization
    return None


def core_headers(headers: Headers) -> dict[str, str | Omit]:
    """Headers for the call to Core: the caller's credential in the same header
    it arrived in, plus any tracing headers.

    Anthropic-specific headers (`anthropic-version`, `anthropic-beta`) are
    deliberately NOT forwarded — neither upstream understands them and they
    must not leak.
    """
    # `omit` suppresses the bearer header the SDK would derive from the
    # placeholder api_key. A real bearer token overwrites it below.
    result: dict[str, str | Omit] = {"Authorization": omit}

    if credential := caller_credential(headers):
        result[credential[0]] = credential[1]

    for name in _TRACE_HEADERS:
        if value := headers.get(name):
            result[name] = value

    return result
