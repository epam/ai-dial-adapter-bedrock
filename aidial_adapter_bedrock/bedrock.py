import json
from abc import ABC
from typing import Any, AsyncIterator, Optional

import boto3
from botocore.eventstream import EventStream
from botocore.response import StreamingBody
from pydantic import BaseModel, Field

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.utils.concurrency import (
    make_async,
    to_async_iterator,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


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

    async def ainvoke_non_streaming(self, model: str, args: dict) -> dict:
        params = self._create_invoke_params(model, args)
        response = await make_async(lambda: self.client.invoke_model(**params))

        log.debug(f"response: {response}")

        body: StreamingBody = response["body"]
        body_dict = json.loads(await make_async(lambda: body.read()))

        log.debug(f"response['body']: {body_dict}")

        return body_dict

    async def ainvoke_streaming(
        self, model: str, args: dict
    ) -> AsyncIterator[dict]:
        params = self._create_invoke_params(model, args)
        response = await make_async(
            lambda: self.client.invoke_model_with_response_stream(**params)
        )

        log.debug(f"response: {response}")

        body: EventStream = response["body"]

        async for event in to_async_iterator(iter(body)):
            chunk = event.get("chunk")
            if chunk:
                chunk_dict = json.loads(chunk.get("bytes").decode())
                log.debug(f"chunk: {chunk_dict}")
                yield chunk_dict


class InvocationMetrics(BaseModel):
    inputTokenCount: int = Field(alias="inputTokenCount")
    outputTokenCount: int = Field(alias="outputTokenCount")
    invocationLatency: int = Field(alias="invocationLatency")
    firstByteLatency: int = Field(alias="firstByteLatency")


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
