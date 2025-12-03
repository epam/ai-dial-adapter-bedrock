from typing import Dict

from aidial_sdk.embeddings import Embeddings, Request, Response

from aidial_adapter_bedrock.adapter_deployments import (
    AdapterEmbeddingsDeployment,
    resolve_upstream,
)
from aidial_adapter_bedrock.deployments import EmbeddingsDeployment
from aidial_adapter_bedrock.llm.model.adapter import get_embeddings_model
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.upstream_config import parse_upstream_config

Deployments = Dict[str, AdapterEmbeddingsDeployment]


class BedrockEmbeddings(Embeddings):
    deployments: Deployments

    def __init__(self, deployments: Deployments) -> None:
        self.deployments = deployments

    @dial_exception_decorator
    async def embeddings(self, request: Request) -> Response:
        deployment = resolve_upstream(EmbeddingsDeployment, request)
        model = await get_embeddings_model(
            deployment=deployment,
            api_key=request.api_key,
            upstream_config=await parse_upstream_config(request),
        )

        return await model.embeddings(request)
