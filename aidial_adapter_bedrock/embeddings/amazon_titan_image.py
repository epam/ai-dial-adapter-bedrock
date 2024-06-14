"""
Amazing Titan Image Embeddings Adapter

See official cookbook for usage instructions:
https://github.com/aws-samples/amazon-bedrock-samples/blob/5752afb78e7fab49cfd42d38bb09d40756bf0ea0/multimodal/Titan/titan-multimodal-embeddings/rag/1_multimodal_rag.ipynb
"""

from typing import List, Optional, Self, Tuple

from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import (
    EmbeddingsRequest,
    EmbeddingsType,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.embeddings.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_bedrock.embeddings.validation import (
    validate_embeddings_request,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class EmbeddingsRequestWithImage(EmbeddingsRequest):
    inputImage: str | None


async def create_request(request: EmbeddingsRequestWithImage) -> dict:
    if isinstance(request.input, str):
        inputText = request.input
    elif len(request.input) == 1:
        inputText = request.input[0]
    else:
        raise ValidationError("Only single text input is supported")

    # This includes all Titan-specific request parameters missing
    # from the OpenAI Embeddings request
    extra_body = request.get_extra_fields()

    return remove_nones(
        {
            "inputText": inputText,
            "inputImage": request.inputImage,
            "dimensions": request.dimensions,
            **extra_body,
        }
    )


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    embedding: List[float]


class AmazonTitanImageEmbeddings(EmbeddingsAdapter):
    model: str
    client: Bedrock

    @classmethod
    def create(cls, client: Bedrock, model: str) -> Self:
        return cls(client=client, model=model)

    async def embeddings(
        self,
        request_body: dict,
        embedding_instruction: Optional[str],
        embedding_type: EmbeddingsType,
    ) -> Tuple[List[List[float]], TokenUsage]:
        request = EmbeddingsRequestWithImage.parse_obj(request_body)

        validate_embeddings_request(
            request,
            embedding_type,
            embedding_instruction,
            [EmbeddingsType.SYMMETRIC],
            supports_dimensions=True,
        )

        request = await create_request(request)
        log.debug(f"request: {request}")

        response_dict = await self.client.ainvoke_non_streaming(
            self.model, request
        )
        response = AmazonResponse.parse_obj(response_dict)

        return [response.embedding], TokenUsage(
            prompt_tokens=response.inputTextTokenCount
        )
