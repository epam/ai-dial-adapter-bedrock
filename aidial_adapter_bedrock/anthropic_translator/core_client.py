import logging

import httpx
from fastapi import Request
from openai import AsyncOpenAI, Omit, omit

from aidial_adapter_bedrock.anthropic_translator.settings import get_api_version
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_CONNECT_TIMEOUT: float = 5.0

_DROPPED_HEADERS: set[str] = {
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

_PLACEHOLDER_API_KEY: str = "-"


async def _log_core_request(request: httpx.Request) -> None:
    if log.isEnabledFor(logging.DEBUG):
        body = await request.aread()
        log.debug(
            "Translated request to Core: %s %s body=%s",
            request.method,
            request.url,
            body.decode("utf-8", errors="replace"),
        )


class _HttpClientPool:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def get(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                event_hooks={"request": [_log_core_request]},
                timeout=httpx.Timeout(
                    _CONNECT_TIMEOUT,
                    read=None,
                    write=None,
                    pool=_CONNECT_TIMEOUT,
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


_http_client_pool: _HttpClientPool = _HttpClientPool()


async def close_http_client() -> None:
    await _http_client_pool.close()


def core_chat_completions_client(base_url: str, deployment: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_PLACEHOLDER_API_KEY,
        base_url=f"{base_url}/openai/deployments/{deployment}",
        max_retries=0,
        default_query={"api-version": get_api_version()},
        http_client=_http_client_pool.get(),
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
    headers["User-Agent"] = (
        "anthropicMessages-to-openaiChatCompletions-translator"
    )
    return headers
