from io import BytesIO
from typing import List, Optional, Tuple, assert_never

from aidial_sdk.chat_completion import (
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
    Role,
)
from aidial_sdk.chat_completion.request import ImageURL
from aidial_sdk.exceptions import RequestValidationError
from PIL import Image
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.resource import (
    AttachmentResource,
    DialResource,
    UnsupportedContentType,
    URLResource,
)
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Attachment, Consumer
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.model.stability.storage import save_to_storage
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.resource import Resource

SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
SUPPORTED_IMAGE_EXTENSIONS = ["jpeg", "jpe", "jpg", "png", "webp"]


async def _download_resource(
    dial_resource: DialResource, storage: FileStorage | None
) -> Resource:
    try:
        return await dial_resource.download(storage)
    except UnsupportedContentType as e:
        raise UserError(
            error_message=f"Unsupported image type: {e.type}",
            usage_message=f"Supported image types: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}",
        )


def _validate_image_size(
    image: Resource,
    width_constraints: Tuple[int, int] | None,
    height_constraints: Tuple[int, int] | None,
) -> None:
    if width_constraints is None and height_constraints is None:
        return

    with Image.open(BytesIO(image.data)) as img:
        width, height = img.size

        for constraints, value, name in [
            (width_constraints, width, "width"),
            (height_constraints, height, "height"),
        ]:
            if constraints is None:
                continue
            min_value, max_value = constraints
            if not (min_value <= value <= max_value):
                error_msg = (
                    f"Image {name} is {value}, but should be "
                    f"between {min_value} and {max_value}"
                )
                raise RequestValidationError(
                    message=error_msg,
                    display_message=error_msg,
                    code="invalid_argument",
                )


def _validate_last_message(messages: List[Message]):
    if not messages:
        raise ValidationError("No messages provided")

    last_message = messages[-1]
    if last_message.role != Role.USER:
        raise ValidationError("Last message must be from user")
    return last_message


class StabilityV2Response(BaseModel):
    seeds: List[int]
    images: List[str]
    # None will indicate that the request was successful
    # Possible values:
    # "Filter reason: prompt"
    # "Filter reason: output image"
    # "Filter reason: input image"
    # "Inference error"
    # null
    finish_reasons: List[Optional[str]]

    def content(self) -> str:
        return " "

    def attachments(self) -> List[Attachment]:
        return [
            Attachment(
                title="Image",
                type="image/png",
                data=image,
            )
            for image in self.images
        ]

    def usage(self) -> TokenUsage:
        return TokenUsage(prompt_tokens=0, completion_tokens=1)

    def throw_if_error(self):
        error = next((reason for reason in self.finish_reasons if reason), None)
        if not error:
            return

        if error == "Inference error":
            raise RuntimeError(error)
        else:
            raise ValidationError(error)


class StabilityV2Adapter(ChatCompletionAdapter):
    model: str
    client: Bedrock
    storage: Optional[FileStorage]
    image_to_image_supported: bool
    width_constraints: Tuple[int, int] | None
    height_constraints: Tuple[int, int] | None

    @classmethod
    def create(
        cls,
        client: Bedrock,
        model: str,
        api_key: str,
        image_to_image_supported: bool,
        image_width_constraints: Tuple[int, int] | None = None,
        image_height_constraints: Tuple[int, int] | None = None,
    ):
        storage: Optional[FileStorage] = create_file_storage(api_key)
        return cls(
            client=client,
            model=model,
            storage=storage,
            image_to_image_supported=image_to_image_supported,
            width_constraints=image_width_constraints,
            height_constraints=image_height_constraints,
        )

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        _validate_last_message(messages)
        return list(range(len(messages) - 1))

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:

        text_prompt = None
        image_resources: List[DialResource] = []
        last_message = _validate_last_message(messages)
        # Handle text content
        match last_message.content:
            case str(text):
                text_prompt = text
            case list():
                text_parts = []

                for part in last_message.content:
                    match part:
                        case MessageContentTextPart(text=text):
                            text_parts.append(text)
                        case MessageContentImagePart(
                            image_url=ImageURL(url=url)
                        ):
                            image_resources.append(
                                URLResource(
                                    url=url,
                                    supported_types=SUPPORTED_IMAGE_TYPES,
                                )
                            )
                        case _:
                            assert_never(part)
                if text_parts:
                    text_prompt = " ".join(text_parts)
            case None:
                pass
            case _:
                assert_never(last_message.content)

        if (
            last_message.custom_content
            and last_message.custom_content.attachments
        ):
            image_resources.extend(
                [
                    AttachmentResource(
                        attachment=attachment,
                        supported_types=SUPPORTED_IMAGE_TYPES,
                    )
                    for attachment in last_message.custom_content.attachments
                ]
            )

        if not self.image_to_image_supported and image_resources:
            raise UserError("Image-to-Image is not supported")
        if len(image_resources) > 1:
            raise UserError("Only one input image is supported")

        if self.image_to_image_supported and image_resources:
            image_resource = await _download_resource(
                image_resources[0], self.storage
            )
            _validate_image_size(
                image_resource, self.width_constraints, self.height_constraints
            )
        else:
            image_resource = None

        if not text_prompt:
            raise UserError("Text prompt is required")

        response, _ = await self.client.ainvoke_non_streaming(
            self.model,
            remove_nones(
                {
                    "prompt": text_prompt,
                    "image": (
                        image_resource.data_base64 if image_resource else None
                    ),
                    "mode": (
                        "image-to-image" if image_resource else "text-to-image"
                    ),
                    "output_format": "png",
                    # This parameter controls how much input image will affect generation from 0 to 1,
                    # where 0 means that output will be identical to input image and 1 means that model will ignore input image
                    # Since there is no recommended default value, we use 0.5 as a middle ground
                    "strength": 0.5 if image_resource else None,
                }
            ),
        )

        stability_response = StabilityV2Response.parse_obj(response)
        stability_response.throw_if_error()

        consumer.append_content(stability_response.content())
        consumer.close_content()

        consumer.add_usage(stability_response.usage())

        for attachment in stability_response.attachments():
            if self.storage:
                attachment = await save_to_storage(self.storage, attachment)
            consumer.add_attachment(attachment)
