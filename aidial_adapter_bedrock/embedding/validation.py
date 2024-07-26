from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_bedrock.llm.errors import ValidationError


def validate_embeddings_request(
    request: EmbeddingsRequest, *, supports_dimensions: bool
) -> None:
    if request.dimensions is not None and not supports_dimensions:
        raise ValidationError("Dimensions parameter is not supported")

    if request.custom_fields:
        if request.custom_fields.instruction is not None:
            raise ValidationError("Instruction prompt is not supported")

        if request.custom_fields.type is not None:
            raise ValidationError(
                "The embedding model does not support embedding types"
            )
