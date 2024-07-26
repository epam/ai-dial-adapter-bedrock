from typing import List, Literal, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class CohereResponse(BaseModel):
    id: str
    response_type: Literal["embeddings_floats"]
    embeddings: List[List[float]]
    texts: List[str]
    # According to https://docs.cohere.com/reference/embed
    # input tokens are expected to be returned in the response field `meta`.
    # However, Bedrock moved it to the response headers.


async def call_embedding_model(
    client: Bedrock, model: str, request: dict
) -> Tuple[List[List[float]], int]:
    body, headers = await client.ainvoke_non_streaming(model, request)
    response = CohereResponse.parse_obj(body)

    input_tokens = int(headers.get("x-amzn-bedrock-input-token-count", "0"))
    if input_tokens == 0:
        log.warning("Can't extract input tokens from embeddings response")

    return response.embeddings, input_tokens
