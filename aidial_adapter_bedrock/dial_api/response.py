from typing import List, Literal

from aidial_sdk.embeddings import Embedding
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from pydantic import BaseModel

from aidial_adapter_bedrock.embedding.encoding import vector_to_base64


class ModelObject(BaseModel):
    object: Literal["model"] = "model"
    id: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]


def _encode_vector(
    encoding_format: Literal["float", "base64"],
    vector: List[float],
) -> List[float] | str:
    base64_encoding = encoding_format == "base64"
    return vector_to_base64(vector) if base64_encoding else vector


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
