from typing import List, Optional, assert_never

from aidial_sdk.chat_completion import (
    Message,
    MessageContentImagePart,
    MessageContentTextPart,
    Role,
)
from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.resource import (
    AttachmentResource,
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
from aidial_adapter_bedrock.llm.model.stabililty.storage import save_to_storage
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from aidial_adapter_bedrock.utils.json import remove_nones

SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]


class StabilityV3Response(BaseModel):
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


class StabilityV3Adapter(ChatCompletionAdapter):
    model: str
    client: Bedrock
    storage: Optional[FileStorage]

    @classmethod
    def create(cls, client: Bedrock, model: str, api_key: str):
        storage: Optional[FileStorage] = create_file_storage(api_key)
        return cls(
            client=client,
            model=model,
            storage=storage,
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
        image_data = None
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
                        case MessageContentImagePart():
                            if image_data is not None:
                                raise ValidationError(
                                    "Only one input image is supported"
                                )
                            resource = await URLResource(
                                url=part.image_url.url,
                                supported_types=SUPPORTED_IMAGE_TYPES,
                            ).download(self.storage)
                            image_data = resource.data_base64
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
            if (
                len(last_message.custom_content.attachments) > 1
                or image_data is not None
            ):
                raise ValidationError("Only one input image is supported")
            resource = await AttachmentResource(
                attachment=last_message.custom_content.attachments[0],
                supported_types=SUPPORTED_IMAGE_TYPES,
            ).download(self.storage)
            image_data = resource.data_base64

        response, _ = await self.client.ainvoke_non_streaming(
            self.model,
            remove_nones(
                {
                    "prompt": text_prompt,
                    "image": image_data,
                    "mode": "image-to-image" if image_data else "text-to-image",
                    "output_format": "png",
                }
            ),
        )

        stability_response = StabilityV3Response.parse_obj(response)
        stability_response.throw_if_error()

        consumer.append_content(stability_response.content())
        consumer.close_content()

        consumer.add_usage(stability_response.usage())

        for attachment in stability_response.attachments():
            if self.storage:
                attachment = await save_to_storage(self.storage, attachment)
            consumer.add_attachment(attachment)
