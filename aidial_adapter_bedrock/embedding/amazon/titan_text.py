"""
Amazing Titan Text Embeddings Adapter

See official cookbook for usage instructions:
https://github.com/aws-samples/amazon-bedrock-samples/blob/5752afb78e7fab49cfd42d38bb09d40756bf0ea0/multimodal/Titan/embeddings/v2/Titan-V2-Embeddings.ipynb
"""

import asyncio
from typing import AsyncIterator, List, Self, Tuple

from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs_without_attachments,
)
from aidial_adapter_bedrock.dial_api.response import make_embeddings_response
from aidial_adapter_bedrock.embedding.amazon.response import (
    call_embedding_model,
)
from aidial_adapter_bedrock.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_bedrock.embedding.validation import (
    validate_embeddings_request,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.json import remove_nones


def create_titan_request(input: str, dimensions: int | None) -> dict:
    return remove_nones({"inputText": input, "dimensions": dimensions})


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
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:

        validate_embeddings_request(
            request,
            supports_type=False,
            supports_dimensions=self.supports_dimensions,
        )

        async def compute_embeddings(
            req: str,
        ) -> Tuple[List[float], int]:
            return await call_embedding_model(
                self.client,
                self.model,
                create_titan_request(req, request.dimensions),
            )

        # NOTE: Amazon Titan doesn't support batched inputs
        tasks = [
            asyncio.create_task(compute_embeddings(sub_request))
            async for sub_request in get_text_inputs(request)
        ]
        results = await asyncio.gather(*tasks)

        return make_embeddings_response(
            model=self.model,
            encoding_format=request.encoding_format,
            vectors=[r[0] for r in results],
            prompt_tokens=sum(r[1] for r in results),
        )
