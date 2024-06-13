from typing import List, Literal, Tuple, TypedDict

from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage


class ModelObject(BaseModel):
    object: Literal["model"] = "model"
    id: str


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelObject]


class EmbeddingsDict(TypedDict):
    index: int
    object: Literal["embedding"]
    embedding: List[float]


class EmbeddingsTokenUsageDict(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponseDict(TypedDict):
    object: Literal["list"]
    model: str
    data: List[EmbeddingsDict]
    usage: EmbeddingsTokenUsageDict


def make_embeddings_response(
    model_id: str, resp: Tuple[List[List[float]], TokenUsage]
) -> EmbeddingsResponseDict:
    vectors, usage = resp

    data: List[EmbeddingsDict] = [
        {"index": idx, "object": "embedding", "embedding": vec}
        for idx, vec in enumerate(vectors)
    ]

    return {
        "object": "list",
        "model": model_id,
        "data": data,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
        },
    }
