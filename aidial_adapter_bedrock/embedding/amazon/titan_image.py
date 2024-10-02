"""
Amazing Titan Image Embeddings Adapter

See official cookbook for usage instructions:
https://github.com/aws-samples/amazon-bedrock-samples/blob/5752afb78e7fab49cfd42d38bb09d40756bf0ea0/multimodal/Titan/titan-multimodal-embeddings/rag/1_multimodal_rag.ipynb
"""

from typing import AsyncIterator, List, Self

from aidial_sdk.chat_completion import Attachment
from aidial_sdk.embeddings import Response as EmbeddingsResponse
from aidial_sdk.embeddings import Usage
from aidial_sdk.embeddings.request import EmbeddingsRequest
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.embedding_inputs import (
    EMPTY_INPUT_LIST_ERROR,
    collect_embedding_inputs,
)
from aidial_adapter_bedrock.dial_api.response import make_embeddings_response
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.embedding.amazon.response import (
    call_embedding_model,
)
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

    def get_image_tokens(self) -> int:
        # According to https://aws.amazon.com/bedrock/pricing/:
        # Price per 1000 input (text) tokens = $0.0008
        # Price per input image = $0.00006
        # Therefore, cost of input image = $0.00006 / ($0.0008 / 1000) = 75 tokens
        return 0 if self.inputImage is None else 75


def create_titan_request(
    request: AmazonRequest, dimensions: int | None
) -> dict:
    conf = None if dimensions is None else {"outputEmbeddingLength": dimensions}
    return remove_nones(
        {
            "inputText": request.inputText,
            "inputImage": request.inputImage,
            "embeddingConfig": conf,
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
    async def on_text(text: str) -> AmazonRequest:
        return AmazonRequest(inputText=text)

    async def on_attachment(attachment: Attachment) -> AmazonRequest:
        return AmazonRequest(
            inputImage=await download_image(attachment, storage)
        )

    async def on_text_or_attachment(text: str | Attachment) -> AmazonRequest:
        if isinstance(text, str):
            return await on_text(text)
        else:
            return await on_attachment(text)

    async def on_mixed(inputs: List[str | Attachment]) -> AmazonRequest:
        if len(inputs) == 0:
            raise EMPTY_INPUT_LIST_ERROR
        elif len(inputs) == 1:
            return await on_text_or_attachment(inputs[0])
        elif len(inputs) == 2:
            if isinstance(inputs[0], str) and isinstance(inputs[1], Attachment):
                return AmazonRequest(
                    inputText=inputs[0],
                    inputImage=await download_image(inputs[1], storage),
                )
            elif isinstance(inputs[0], Attachment) and isinstance(
                inputs[1], str
            ):
                return AmazonRequest(
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
    def create(cls, client: Bedrock, model: str, api_key: str) -> Self:
        storage = create_file_storage(api_key)
        return cls(client=client, model=model, storage=storage)

    async def embeddings(
        self, request: EmbeddingsRequest
    ) -> EmbeddingsResponse:

        validate_embeddings_request(
            request,
            supports_type=False,
            supports_dimensions=True,
        )

        vectors: List[List[float] | str] = []
        token_count = 0

        # NOTE: Amazon Titan doesn't support batched inputs
        async for sub_request in get_requests(request, self.storage):
            embedding, text_tokens = await call_embedding_model(
                self.client,
                self.model,
                create_titan_request(sub_request, request.dimensions),
            )

            image_tokens = sub_request.get_image_tokens()

            vector = (
                vector_to_base64(embedding)
                if request.encoding_format == "base64"
                else embedding
            )

            vectors.append(vector)
            token_count += text_tokens + image_tokens

        return make_embeddings_response(
            model=self.model,
            vectors=vectors,
            usage=Usage(prompt_tokens=token_count, total_tokens=token_count),
        )
