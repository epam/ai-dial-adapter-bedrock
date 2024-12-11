from typing import List, Literal, Self

from aidial_sdk.embeddings import Embedding
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from pydantic import BaseModel

from aidial_adapter_bedrock.embedding.encoding import vector_to_base64


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


def _encode_vector(
    encoding_format: Literal["float", "base64"], vector: List[float]
) -> List[float] | str:
    return vector_to_base64(vector) if encoding_format == "base64" else vector


def make_embeddings_response(
    model: str,
    encoding_format: Literal["float", "base64"],
    vectors: List[List[float]],
    prompt_tokens: int,
) -> EmbeddingsResponse:

    embeddings = [_encode_vector(encoding_format, v) for v in vectors]

    data: List[Embedding] = [
        Embedding(index=index, embedding=embedding)
        for index, embedding in enumerate(embeddings)
    ]

    return EmbeddingsResponse(
        model=model,
        data=data,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        ),
    )
