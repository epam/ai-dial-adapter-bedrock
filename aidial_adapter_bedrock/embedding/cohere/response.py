from typing import Literal

from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock, prompt_tokens_from_headers


class CohereResponse(BaseModel):
    id: str
    response_type: Literal["embeddings_floats"]
    embeddings: list[list[float]]
    texts: list[str]
    # According to https://docs.cohere.com/reference/embed
    # input tokens are expected to be returned in the response field `meta`.
    # However, Bedrock moved it to the response headers.


async def call_embedding_model(
    client: Bedrock, model: str, request: dict
) -> tuple[list[list[float]], int]:
    body, headers = await client.ainvoke_non_streaming(model, request)
    response = CohereResponse.model_validate(body)

    input_tokens = prompt_tokens_from_headers(headers) or 0
    return response.embeddings, input_tokens
