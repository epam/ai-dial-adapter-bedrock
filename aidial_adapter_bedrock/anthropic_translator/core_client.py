from fastapi import Request
from openai import AsyncOpenAI, Omit, omit

from aidial_adapter_bedrock.bedrock import get_anthropic_http_client

_DROPPED_HEADERS = {
    "x-upstream-endpoint",
    "x-upstream-key",
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "x-dial-deployment-id",
    "x-dial-deployment-features",
    "x-dial-override-name",
    "x-dial-cache-breakpoint-path",
    "host",
    "content-length",
    "content-type",
    "content-encoding",
    "accept",
    "accept-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "user-agent",
}


def core_chat_completions_client(dial_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        # Required by the SDK; Core authenticates the forwarded `api-key` header.
        api_key="-",
        base_url=f"{dial_url}/openai/v1",
        max_retries=0,
        http_client=get_anthropic_http_client(),
    )


def core_headers(request: Request) -> dict[str, str | Omit]:
    headers: dict[str, str | Omit] = {"Authorization": omit}
    headers.update(
        {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _DROPPED_HEADERS
        }
    )
    return headers
