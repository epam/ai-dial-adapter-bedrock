from aidial_sdk.embeddings import Embeddings, Request, Response

from aidial_adapter_bedrock.adapter_deployments import (
    AdapterEmbeddingsDeployment,
)
from aidial_adapter_bedrock.aws_client_config import AWSClientConfigFactory
from aidial_adapter_bedrock.llm.model.adapter import get_embeddings_model
from aidial_adapter_bedrock.server.exceptions import dial_exception_decorator


class BedrockEmbeddings(Embeddings):
    deployment: AdapterEmbeddingsDeployment

    def __init__(self, deployment: AdapterEmbeddingsDeployment) -> None:
        self.deployment = deployment

    @dial_exception_decorator
    async def embeddings(self, request: Request) -> Response:
        aws_client_config = AWSClientConfigFactory(request).get_client_config()

        model = await get_embeddings_model(
            deployment=self.deployment,
            api_key=request.api_key,
            aws_client_config=aws_client_config,
        )

        return await model.embeddings(request)
