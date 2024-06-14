"""
Amazing Titan Text Embeddings Adapter

See official cookbook for usage instructions:
https://github.com/aws-samples/amazon-bedrock-samples/blob/5752afb78e7fab49cfd42d38bb09d40756bf0ea0/multimodal/Titan/embeddings/v2/Titan-V2-Embeddings.ipynb
"""

from typing import Iterable, List, Optional, Self, Tuple

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
from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def create_requests(request: EmbeddingsRequest) -> Iterable[dict]:
    inputs: List[str] = (
        [request.input] if isinstance(request.input, str) else request.input
    )

    # This includes all Titan-specific request parameters missing
    # from the OpenAI Embeddings request, e.g. "normalize" boolean flag.
    extra_body = request.get_extra_fields()

    # NOTE: Amazon Titan doesn't support batched inputs
    for input in inputs:
        yield remove_nones(
            {
                "inputText": input,
                "dimensions": request.dimensions,
                **extra_body,
            }
        )


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    embedding: List[float]


class AmazonTitanTextEmbeddings(EmbeddingsAdapter):
    model: str
    client: Bedrock
    supports_dimensions: bool

    @classmethod
    def create(
        cls, client: Bedrock, model: str, supports_dimensions: bool
    ) -> Self:
        return cls(
            client=client, model=model, supports_dimensions=supports_dimensions
        )

    async def embeddings(
        self,
        request_body: dict,
        embedding_instruction: Optional[str],
        embedding_type: EmbeddingsType,
    ) -> Tuple[List[List[float]], TokenUsage]:
        request = EmbeddingsRequest.parse_obj(request_body)

        validate_embeddings_request(
            request,
            embedding_type,
            embedding_instruction,
            [EmbeddingsType.SYMMETRIC],
            self.supports_dimensions,
        )

        embeddings: List[List[float]] = []
        usage = TokenUsage()

        for request in create_requests(request):
            log.debug(f"request: {request}")

            response_dict = await self.client.ainvoke_non_streaming(
                self.model, request
            )
            response = AmazonResponse.parse_obj(response_dict)
            embeddings.append(response.embedding)
            usage.prompt_tokens += response.inputTextTokenCount

        return embeddings, usage
