from abc import ABC, abstractmethod

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel


class EmbeddingsAdapter(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        pass
