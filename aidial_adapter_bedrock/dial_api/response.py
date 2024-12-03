from typing import List, Literal, Self

from aidial_sdk.embeddings import Embedding
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from pydantic import BaseModel


class Capabilities(BaseModel):
    chat_completion: bool = False
    completion: bool = False
    embeddings: bool = False
    fine_tune: bool = False
    inference: bool = False


class ModelObject(BaseModel):
    object: Literal["model"] = "model"
    capabilities: Capabilities = Capabilities()
    id: str

    @classmethod
    def chat_completions(cls, id: str) -> Self:
        return cls(id=id, capabilities=Capabilities(chat_completion=True))

    @classmethod
    def embeddings(cls, id: str) -> Self:
        return cls(id=id, capabilities=Capabilities(embeddings=True))


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
