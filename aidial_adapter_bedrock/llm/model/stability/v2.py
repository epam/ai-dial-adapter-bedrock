from typing import List, Optional, assert_never

from aidial_sdk.chat_completion import (
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
    Role,
)
from aidial_sdk.chat_completion.request import ImageURL
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.resource import (
    AttachmentResource,
    DialResource,
    URLResource,
)
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Attachment, Consumer
from aidial_adapter_bedrock.llm.errors import ValidationError
from aidial_adapter_bedrock.llm.model.stability.storage import save_to_storage
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from aidial_adapter_bedrock.utils.json import remove_nones

SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]


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

    @classmethod
    def create(
        cls,
        client: Bedrock,
        model: str,
        api_key: str,
        image_to_image_supported: bool,
    ):
        storage: Optional[FileStorage] = create_file_storage(api_key)
        return cls(
            client=client,
            model=model,
            storage=storage,
            image_to_image_supported=image_to_image_supported,
        )

    def _validate_last_message(self, messages: List[Message]):
        if not messages:
            raise ValidationError("No messages provided")

        last_message = messages[-1]
        if last_message.role != Role.USER:
            raise ValidationError("Last message must be from user")
        return last_message

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        self._validate_last_message(messages)
        return list(range(len(messages) - 1))

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:

        text_prompt = None
        image_resources: List[DialResource] = []
        last_message = self._validate_last_message(messages)
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
            raise ValidationError(
                f"Image-to-image is not supported for {self.model}"
            )
        if len(image_resources) > 1:
            raise ValidationError("Only one input image is supported")

        response, _ = await self.client.ainvoke_non_streaming(
            self.model,
            remove_nones(
                {
                    "prompt": text_prompt,
                    "image": (
                        (
                            await image_resources[0].download(self.storage)
                        ).data_base64
                        if image_resources
                        else None
                    ),
                    "mode": (
                        "image-to-image" if image_resources else "text-to-image"
                    ),
                    "output_format": "png",
                    # This parameter controls how much input image will affect generation from 0 to 1,
                    # where 0 means that output will be identical to input image and 1 means that model will ignore input image
                    # Since there is no recommended default value, we use 0.5 as a middle ground
                    "strength": 0.5 if image_resources else None,
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
