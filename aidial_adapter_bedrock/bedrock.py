import asyncio
import json
from typing import Any, AsyncIterator

import boto3

from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class Bedrock:
    client: Any

    def __init__(self, client: Any):
        self.client = client

    @classmethod
    async def acreate(cls, region: str) -> "Bedrock":
        client = await make_async(
            lambda _: boto3.Session().client("bedrock-runtime", region), ()
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
        response = await make_async(
            lambda _: self.client.invoke_model(**params), ()
        )

        body = json.loads(response["body"].read())
        log.debug(f"body [stream=false]: {args}")
        return body

    async def ainvoke_streaming(
        self, model: str, args: dict
    ) -> AsyncIterator[dict]:
        params = self._create_invoke_params(model, args)
        response = await make_async(
            lambda _: self.client.invoke_model_with_response_stream(**params),
            (),
        )

        body = response["body"]
        for event in body:
            chunk = event.get("chunk")
            if chunk:
                chunk_dict = json.loads(chunk.get("bytes").decode())
                log.debug(f"chunk [stream=true]: {chunk_dict}")
                yield chunk_dict
                await asyncio.sleep(1e-6)
