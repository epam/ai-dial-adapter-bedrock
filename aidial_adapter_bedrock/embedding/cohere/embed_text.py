"""
Text Embeddings Adapter for Cohere Embed model

See the documentation:
https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
https://docs.cohere.com/reference/embed
"""

from typing import AsyncIterator, List, Self

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs_without_attachments,
)
from aidial_adapter_bedrock.dial_api.response import make_embeddings_response
from aidial_adapter_bedrock.embedding.cohere.response import (
    call_embedding_model,
)
from aidial_adapter_bedrock.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_bedrock.embedding.encoding import vector_to_base64
from aidial_adapter_bedrock.embedding.validation import (
    validate_embeddings_request,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.json import remove_nones


def create_cohere_request(texts: List[str], input_type: str) -> dict:
    return remove_nones(
        {
            "texts": texts,
            "input_type": input_type,
        }
    )


def get_text_inputs(request: EmbeddingsRequest) -> AsyncIterator[str]:
    async def on_texts(texts: List[str]) -> str:
        if len(texts) == 0:
            raise EMPTY_INPUT_LIST_ERROR
        elif len(texts) == 1:
            return texts[0]
        else:
            raise ValidationError(
                "No more than one element is allowed in an element of custom_input list"
            )

    return collect_embedding_inputs_without_attachments(
        request, on_texts=on_texts
    )


class CohereTextEmbeddings(EmbeddingsAdapter):
    model: str
    client: Bedrock

    @classmethod
    def create(cls, client: Bedrock, model: str) -> Self:
        return cls(client=client, model=model)

    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:

        validate_embeddings_request(
            request,
            supports_type=True,
            supports_dimensions=False,
        )

        input_type: str | None = (
            request.custom_fields and request.custom_fields.type
        )

        if input_type is None:
            raise ValidationError(
                "Embedding type request parameter is required"
            )

        text_inputs = [txt async for txt in get_text_inputs(request)]

        embedding_request = create_cohere_request(text_inputs, input_type)

        embeddings, tokens = await call_embedding_model(
            self.client, self.model, embedding_request
        )

        vectors: List[List[float] | str] = [
            (
                vector_to_base64(embedding)
                if request.encoding_format == "base64"
                else embedding
            )
            for embedding in embeddings
        ]

        return make_embeddings_response(
            model=self.model,
            vectors=vectors,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )
