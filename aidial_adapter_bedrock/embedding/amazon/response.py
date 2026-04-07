from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    embedding: list[float]


async def call_embedding_model(
    client: Bedrock, model: str, request: dict
) -> tuple[list[float], int]:
    response_dict, _headers = await client.ainvoke_non_streaming(model, request)
    response = AmazonResponse.model_validate(response_dict)
    return response.embedding, response.inputTextTokenCount
