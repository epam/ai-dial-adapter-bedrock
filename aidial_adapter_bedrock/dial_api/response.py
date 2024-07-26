from typing import List, Literal

from aidial_sdk.embeddings import Embedding
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from pydantic import BaseModel


class ModelObject(BaseModel):
    object: Literal["model"] = "model"
    id: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]


def make_embeddings_response(
    model: str, vectors: List[List[float] | str], usage: Usage
) -> EmbeddingsResponse:

    data: List[Embedding] = [
        Embedding(index=index, embedding=embedding)
        for index, embedding in enumerate(vectors)
    ]

    return EmbeddingsResponse(model=model, data=data, usage=usage)
