from typing import Iterable, List, Literal, Optional, Self, Tuple

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
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


def validate_parameters(
    encoding_format: Literal["float", "base64"],
    embedding_type: EmbeddingsType,
    embedding_instruction: Optional[str],
    supported_embedding_types: List[EmbeddingsType],
) -> None:
    if encoding_format == "base64":
        raise ValidationError("Base64 encoding format is not supported")

    if embedding_instruction is not None:
        raise ValidationError("Instruction prompt is not supported")

    assert (
        len(supported_embedding_types) != 0
    ), "The embedding model doesn't support any embedding types"

    if embedding_type not in supported_embedding_types:
        allowed = ", ".join([e.value for e in supported_embedding_types])
        raise ValidationError(
            f"Embedding types other than {allowed} are not supported"
        )


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

    @classmethod
    def create(cls, client: Bedrock, model: str) -> Self:
        return cls(client=client, model=model)

    async def embeddings(
        self,
        request_body: dict,
        embedding_instruction: Optional[str],
        embedding_type: EmbeddingsType,
    ) -> Tuple[List[List[float]], TokenUsage]:
        request = EmbeddingsRequest.parse_obj(request_body)

        validate_parameters(
            request.encoding_format,
            embedding_type,
            embedding_instruction,
            [EmbeddingsType.SYMMETRIC],
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
