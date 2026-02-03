from abc import ABC, abstractmethod

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest


class EmbeddingsAdapter(ABC):
    @abstractmethod
    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        pass
