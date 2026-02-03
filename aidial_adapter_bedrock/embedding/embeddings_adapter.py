from abc import ABC, abstractmethod

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel, ConfigDict


class EmbeddingsAdapter(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:
        pass
