from typing import List, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    embedding: List[float]


async def call_embedding_model(
    client: Bedrock, model: str, request: dict
) -> Tuple[List[float], int]:
    log.debug(f"request: {request}")
    response_dict = await client.ainvoke_non_streaming(model, request)
    response = AmazonResponse.parse_obj(response_dict)
    return response.embedding, response.inputTextTokenCount
