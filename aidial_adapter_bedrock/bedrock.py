import json
from abc import ABC
from datetime import datetime
from functools import cache
from logging import DEBUG
from typing import Any, AsyncIterator, Mapping, Optional, Tuple, Unpack

import anthropic
import boto3
import botocore
import httpx
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from botocore.eventstream import EventStream
from botocore.response import StreamingBody
from pydantic import BaseModel, Field

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.converse.types import ConverseRequest
from aidial_adapter_bedrock.upstream_config import (
    ApiKeyUpstreamConfig,
    CloudUpstreamConfig,
    UpstreamConfig,
)
from aidial_adapter_bedrock.utils.cache import ttl_cache
from aidial_adapter_bedrock.utils.concurrency import (
    make_async,
    to_async_iterator,
)
from aidial_adapter_bedrock.utils.env import get_env_int
from aidial_adapter_bedrock.utils.json import json_dumps_short
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

Body = dict
Headers = Mapping[str, str]

ANTHROPIC_MAX_CONNECTIONS = get_env_int("ANTHROPIC_MAX_CONNECTIONS", 1000)
ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS = get_env_int(
    "ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS", 100
)
BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS = get_env_int(
    "BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS", 1000
)


@cache
def get_default_anthropic_timeout() -> httpx.Timeout:
    # Providing a timeout marginally different from the default Anthropic timeout
    # in order to disable the check that throws an error when
    # stream=False & max_tokens>=128K/6:
    # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176

    timeout = anthropic._constants.DEFAULT_TIMEOUT.as_dict()
    timeout["connect"] *= 1.0001  # type: ignore
    return httpx.Timeout(**timeout)


@ttl_cache
async def create_anthropic_client(
    upstream_config: UpstreamConfig,
) -> Tuple[datetime | None, AsyncAnthropicBedrock | AsyncAnthropic]:
    # NOTE: default connection limits set by the anthropic library:
    # * max_connections=1000
    # * max_keepalive_connections=100
    # Meaning that there couldn't be more than 1000 concurrent requests.
    # Anything beyond will be blocked.
    http_client = httpx.AsyncClient(
        timeout=get_default_anthropic_timeout(),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=ANTHROPIC_MAX_CONNECTIONS,
            max_keepalive_connections=ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS,
        ),
    )

    if isinstance(upstream_config, ApiKeyUpstreamConfig):
        anthropic_client = AsyncAnthropic(
            api_key=upstream_config.api_key,
            http_client=http_client,
        )
        return (None, anthropic_client)
    else:
        (expiration, creds) = await upstream_config.get_credentials()
        anthropic_client = AsyncAnthropicBedrock(
            aws_region=upstream_config.region,
            aws_access_key=creds.aws_access_key_id,
            aws_secret_key=creds.aws_secret_access_key,
            aws_session_token=creds.aws_session_token,
            http_client=http_client,
        )
        return expiration, anthropic_client


@ttl_cache
async def create_boto_client(
    service_name: str, upstream_config: CloudUpstreamConfig
) -> Tuple[datetime | None, Any]:

    (expiration, creds) = await upstream_config.get_credentials()

    # NOTE: max number of connections to a single host that are being persisted.
    # Greater number of connections do not block each other.
    config = botocore.client.Config(  # type: ignore
        max_pool_connections=BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS
    )

    # NOTE: Session isn't thread-safe, but client is.
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html#caveats
    client = await make_async(
        lambda: boto3.Session().client(
            service_name,
            region_name=upstream_config.region,
            aws_access_key_id=creds.aws_access_key_id,
            aws_secret_access_key=creds.aws_secret_access_key,
            aws_session_token=creds.aws_session_token,
            config=config,
        )
    )
    return (expiration, client)


class Bedrock:
    client: Any

    def __init__(self, client: Any):
        self.client = client

    @classmethod
    async def acreate(cls, upstream_config: UpstreamConfig) -> "Bedrock":
        if isinstance(upstream_config, ApiKeyUpstreamConfig):
            raise ValueError(
                "Authentication via API key isn't supported for the deployment"
            )
        client = await create_boto_client("bedrock-runtime", upstream_config)
        return cls(client)

    async def aconverse_non_streaming(
        self, model: str, **params: Unpack[ConverseRequest]
    ):
        response = await make_async(
            lambda: self.client.converse(modelId=model, **params)
        )
        return response

    async def aconverse_streaming(
        self, model: str, **params: Unpack[ConverseRequest]
    ):
        response = await make_async(
            lambda: self.client.converse_stream(modelId=model, **params)
        )

        return to_async_iterator(iter(response["stream"]))

    def _create_invoke_params(self, model: str, body: dict) -> dict:
        return {
            "modelId": model,
            "body": json.dumps(body),
            "accept": "application/json",
            "contentType": "application/json",
        }

    async def ainvoke_non_streaming(
        self, model: str, args: dict
    ) -> Tuple[Body, Headers]:

        if log.isEnabledFor(DEBUG):
            log.debug(
                f"request: {json_dumps_short({'model': model, 'args': args})}"
            )

        params = self._create_invoke_params(model, args)
        response = await make_async(lambda: self.client.invoke_model(**params))

        if log.isEnabledFor(DEBUG):
            log.debug(f"response: {json_dumps_short(response)}")

        body: StreamingBody = response["body"]
        body_dict = json.loads(await make_async(lambda: body.read()))

        response_headers = response.get("ResponseMetadata", {}).get(
            "HTTPHeaders", {}
        )

        if log.isEnabledFor(DEBUG):
            log.debug(f"response['body']: {json_dumps_short(body_dict)}")

        return body_dict, response_headers

    async def ainvoke_streaming(
        self, model: str, args: dict
    ) -> AsyncIterator[dict]:
        if log.isEnabledFor(DEBUG):
            log.debug(
                f"request: {json_dumps_short({'model': model, 'args': args})}"
            )

        params = self._create_invoke_params(model, args)
        response = await make_async(
            lambda: self.client.invoke_model_with_response_stream(**params)
        )

        if log.isEnabledFor(DEBUG):
            log.debug(f"response: {json_dumps_short(response)}")

        body: EventStream = response["body"]

        async for event in to_async_iterator(iter(body)):
            chunk = event.get("chunk")
            if chunk:
                chunk_dict = json.loads(chunk.get("bytes").decode())
                if log.isEnabledFor(DEBUG):
                    log.debug(f"chunk: {json_dumps_short(chunk_dict)}")
                yield chunk_dict


class InvocationMetrics(BaseModel):
    inputTokenCount: int
    outputTokenCount: int
    invocationLatency: int
    firstByteLatency: int


class ResponseWithInvocationMetricsMixin(ABC, BaseModel):
    invocation_metrics: Optional[InvocationMetrics] = Field(
        alias="amazon-bedrock-invocationMetrics"
    )

    def usage_from_metrics(self) -> TokenUsage:
        metrics = self.invocation_metrics
        if metrics is None:
            return TokenUsage()

        return TokenUsage(
            prompt_tokens=metrics.inputTokenCount,
            completion_tokens=metrics.outputTokenCount,
        )


def prompt_tokens_from_headers(headers: Headers) -> int | None:
    try:
        return int(headers["x-amzn-bedrock-input-token-count"])
    except Exception:
        log.error(
            "Failed to extract prompt token usage from the response headers"
        )
        return None


def completion_tokens_from_headers(headers: Headers) -> int | None:
    try:
        return int(headers["x-amzn-bedrock-output-token-count"])
    except Exception:
        log.error(
            "Failed to extract completion token usage from the response headers"
        )
        return None


def usage_from_headers(response_headers: Headers) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt_tokens_from_headers(response_headers) or 0,
        completion_tokens=completion_tokens_from_headers(response_headers) or 0,
    )
