from aidial_sdk.embeddings import Embeddings, Request, Response

from aidial_adapter_bedrock.deployments import EmbeddingsDeployment
from aidial_adapter_bedrock.llm.model.adapter import get_embeddings_model
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator
from aidial_adapter_bedrock.upstream_config import parse_upstream_config
from aidial_adapter_bedrock.utils.adapter_deployments import (
    resolve_upstream_deployment_id_from_request,
)


class BedrockEmbeddings(Embeddings):
    @dial_exception_decorator
    async def embeddings(self, request: Request) -> Response:
        deployment = resolve_upstream_deployment_id_from_request(
            EmbeddingsDeployment, request
        )
        model = await get_embeddings_model(
            deployment=deployment,
            api_key=request.api_key,
            upstream_config=await parse_upstream_config(
                request.original_request
            ),
        )

        return await model.embeddings(request)
