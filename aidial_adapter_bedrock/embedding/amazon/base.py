from typing import List, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    embedding: List[float]


async def call_embedding_model(
    client: Bedrock, model: str, request: dict
) -> Tuple[List[float], int]:
    response_dict = await client.ainvoke_non_streaming(model, request)
    response = AmazonResponse.parse_obj(response_dict)
    return response.embedding, response.inputTextTokenCount
