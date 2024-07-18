from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_bedrock.llm.errors import ValidationError


def validate_embeddings_request(
    request: EmbeddingsRequest, *, supports_dimensions: bool
) -> None:
    if request.encoding_format == "base64":
        raise ValidationError("Base64 encoding format is not supported")

    if request.dimensions is not None and not supports_dimensions:
        raise ValidationError("Dimensions parameter isn't supported")

    if request.custom_fields:
        if request.custom_fields.instruction is not None:
            raise ValidationError("Instruction prompt is not supported")

        if request.custom_fields.type is not None:
            raise ValidationError(
                "The embedding model doesn't support any embedding types"
            )
