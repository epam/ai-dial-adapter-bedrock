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

    def _create_args(self, model: str, body: dict) -> dict:
        return {
            "modelId": model,
            "body": json.dumps(body),
            "accept": "application/json",
            "contentType": "application/json",
        }

    async def ainvoke_non_streaming(self, model: str, body: dict) -> dict:
        args = self._create_args(model, body)
        response = await make_async(
            lambda _: self.client.invoke_model(**args), ()
        )

        body = json.loads(response["body"].read())
        log.debug(f"body [stream=false]: {body}")
        return body

    async def ainvoke_streaming(
        self, model: str, body: dict
    ) -> AsyncIterator[dict]:
        args = self._create_args(model, body)
        response = await make_async(
            lambda _: self.client.invoke_model_with_response_stream(**args), ()
        )

        body = response["body"]
        for event in body:
            chunk = event.get("chunk")
            if chunk:
                chunk_dict = json.loads(chunk.get("bytes").decode())
                log.debug(f"chunk [stream=true]: {chunk_dict}")

                yield chunk_dict
