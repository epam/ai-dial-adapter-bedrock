import json
from abc import ABC
from asyncio import Lock
from collections import defaultdict
from logging import DEBUG
from typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Mapping,
    Optional,
    Tuple,
    Unpack,
)

import boto3
import botocore
from botocore.eventstream import EventStream
from botocore.response import StreamingBody
from pydantic import BaseModel, Field

from aidial_adapter_bedrock.aws_client_config import AWSClientConfig
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.converse.types import ConverseRequest
from aidial_adapter_bedrock.utils.concurrency import (
    make_async,
    to_async_iterator,
)
from aidial_adapter_bedrock.utils.json import json_dumps_short
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

Body = dict
Headers = Mapping[str, str]


class _BedrockClientFactory:
    _clients: ClassVar[dict[str, Any]] = {}
    _locks: ClassVar[dict[str, Lock]] = defaultdict(Lock)

    # FIX add TTL?
    @classmethod
    async def get_client(cls, client_config: dict) -> Any:
        key = json.dumps(client_config, sort_keys=True, separators=(",", ":"))

        async with cls._locks[key]:
            if client := cls._clients.get(key):
                log.debug(f"Bedrock client cache-hit: {key}")
                return client

            log.debug(f"Bedrock client cache-miss: {key}")
            config = botocore.client.Config(max_pool_connections=10)  # type: ignore

            client = await make_async(
                lambda: boto3.Session().client(**client_config, config=config)
            )
            cls._clients[key] = client
            return client


class Bedrock:
    client: Any

    def __init__(self, client: Any):
        self.client = client

    @classmethod
    async def acreate(cls, aws_client_config: AWSClientConfig) -> "Bedrock":
        client_kwargs = aws_client_config.get_boto_client_kwargs()
        client_kwargs["service_name"] = "bedrock-runtime"
        client = await _BedrockClientFactory.get_client(client_kwargs)
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
