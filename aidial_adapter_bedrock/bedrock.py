import json
from abc import ABC
from logging import DEBUG
from typing import Any, AsyncIterator, Mapping, Optional, Tuple

import boto3
from botocore.eventstream import EventStream
from botocore.response import StreamingBody
from pydantic import BaseModel, Field

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.utils.concurrency import (
    make_async,
    to_async_iterator,
)
from aidial_adapter_bedrock.utils.json import json_dumps_short
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

Body = dict
Headers = Mapping[str, str]


class Bedrock:
    client: Any

    def __init__(self, client: Any):
        self.client = client

    @classmethod
    async def acreate(cls, region: str) -> "Bedrock":
        client = await make_async(
            lambda: boto3.Session().client("bedrock-runtime", region)
        )
        return cls(client)

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

    def usage_by_metrics(self) -> TokenUsage:
        metrics = self.invocation_metrics
        if metrics is None:
            return TokenUsage()

        return TokenUsage(
            prompt_tokens=metrics.inputTokenCount,
            completion_tokens=metrics.outputTokenCount,
        )
