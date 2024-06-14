from typing import List, Optional

from aidial_adapter_bedrock.dial_api.request import (
    EmbeddingsRequest,
    EmbeddingsType,
)
from aidial_adapter_bedrock.llm.errors import ValidationError


def validate_embeddings_request(
    request: EmbeddingsRequest,
    embedding_type: EmbeddingsType,
    embedding_instruction: Optional[str],
    supported_embedding_types: List[EmbeddingsType],
    supports_dimensions: bool,
) -> None:
    if request.encoding_format == "base64":
        raise ValidationError("Base64 encoding format is not supported")

    if request.dimensions is not None and not supports_dimensions:
        raise ValidationError("Dimensions parameter isn't supported")

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
