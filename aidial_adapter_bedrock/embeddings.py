from aidial_sdk.embeddings import Embeddings, Request, Response

from aidial_adapter_bedrock.deployments import EmbeddingsDeployment
from aidial_adapter_bedrock.llm.model.adapter import get_embeddings_model
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator


class BedrockEmbeddings(Embeddings):
    def __init__(self, region: str):
        self.region = region

    @dial_exception_decorator
    async def embeddings(self, request: Request) -> Response:

        model = await get_embeddings_model(
            deployment=EmbeddingsDeployment(request.deployment_id),
            region=self.region,
            headers=request.headers,
        )

        return await model.embeddings(request)
