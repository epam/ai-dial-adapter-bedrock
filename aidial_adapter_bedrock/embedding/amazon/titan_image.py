"""
Amazing Titan Image Embeddings Adapter

See official cookbook for usage instructions:
https://github.com/aws-samples/amazon-bedrock-samples/blob/5752afb78e7fab49cfd42d38bb09d40756bf0ea0/multimodal/Titan/titan-multimodal-embeddings/rag/1_multimodal_rag.ipynb
"""

from typing import AsyncIterator, List, Mapping, Self

from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.embedding_inputs import (
    collect_embedding_inputs,
)
from aidial_adapter_bedrock.dial_api.response import make_embeddings_response
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.embedding.amazon.base import call_embedding_model
from aidial_adapter_bedrock.embedding.attachments import download_base64_data
from aidial_adapter_bedrock.embedding.embeddings_adapter import (
    EmbeddingsAdapter,
)
from aidial_adapter_bedrock.embedding.encoding import vector_to_base64
from aidial_adapter_bedrock.embedding.validation import (
    validate_embeddings_request,
)
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.utils.json import remove_nones


class AmazonRequest(BaseModel):
    inputText: str | None = None
    inputImage: str | None = None


def create_titan_request(
    request: AmazonRequest, dimensions: int | None
) -> dict:
    return remove_nones(
        {
            "inputText": request.inputText,
            "inputImage": request.inputImage,
            "dimensions": dimensions,
        }
    )


async def download_image(
    attachment: Attachment, storage: FileStorage | None
) -> str:
    _content_type, data = await download_base64_data(
        attachment, storage, ["image/png"]
    )
    return data


def get_requests(
    request: EmbeddingsRequest, storage: FileStorage | None
) -> AsyncIterator[AmazonRequest]:
    async def on_text(text: str):
        return AmazonRequest(inputText=text)

    async def on_attachment(attachment: Attachment):
        return AmazonRequest(
            inputImage=await download_image(attachment, storage)
        )

    async def on_text_or_attachment(text: str | Attachment):
        if isinstance(text, str):
            return await on_text(text)
        else:
            return await on_attachment(text)

    async def on_mixed(
        inputs: List[str | Attachment],
    ) -> AsyncIterator[AmazonRequest]:
        if len(inputs) == 0:
            pass
        elif len(inputs) == 1:
            yield await on_text_or_attachment(inputs[0])
        elif len(inputs) == 2:
            if isinstance(inputs[0], str) and isinstance(inputs[1], Attachment):
                yield AmazonRequest(
                    inputText=inputs[0],
                    inputImage=await download_image(inputs[1], storage),
                )
            elif isinstance(inputs[0], Attachment) and isinstance(
                inputs[1], str
            ):
                yield AmazonRequest(
                    inputText=inputs[1],
                    inputImage=await download_image(inputs[0], storage),
                )
            else:
                raise ValidationError(
                    "The first element of a custom_input list element must be a string and the second element must be an image attachment or vice versa"
                )
        else:
            raise ValidationError(
                "No more than two elements are allowed in an element of custom_input list"
            )

    return collect_embedding_inputs(
        request,
        on_text=on_text,
        on_attachment=on_attachment,
        on_mixed=on_mixed,
    )


class AmazonResponse(BaseModel):
    inputTextTokenCount: int
    embedding: List[float]


class AmazonTitanImageEmbeddings(EmbeddingsAdapter):
    model: str
    client: Bedrock
    storage: FileStorage | None

    @classmethod
    def create(
        cls, client: Bedrock, model: str, headers: Mapping[str, str]
    ) -> Self:
        storage: FileStorage | None = create_file_storage(headers)
        return cls(client=client, model=model, storage=storage)

    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:

        # The model in fact does not support dimensions,
        # but the documentation claims it does
        validate_embeddings_request(request, supports_dimensions=False)

        vectors: List[List[float] | str] = []
        token_count = 0

        # NOTE: Amazon Titan doesn't support batched inputs
        async for text_input in get_requests(request, self.storage):
            sub_request = create_titan_request(text_input, request.dimensions)
            embedding, tokens = await call_embedding_model(
                self.client, self.model, sub_request
            )

            vector = (
                vector_to_base64(embedding)
                if request.encoding_format == "base64"
                else embedding
            )

            vectors.append(vector)
            token_count += tokens

        return make_embeddings_response(
            model=self.model,
            vectors=vectors,
            usage=Usage(prompt_tokens=token_count, total_tokens=token_count),
        )