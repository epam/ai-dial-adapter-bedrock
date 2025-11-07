from io import BytesIO
from typing import List, Literal, Optional, Tuple, Type

from aidial_sdk.chat_completion import Attachment, Message
from aidial_sdk.exceptions import (
    InternalServerError,
    InvalidRequestError,
    RequestValidationError,
)
from PIL import Image
from pydantic import BaseModel
from typing_extensions import assert_never

from aidial_adapter_bedrock.adapter_deployments import AdapterDeployment
from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.resource import (
    DialResource,
    UnsupportedContentType,
)
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.errors import UserError
from aidial_adapter_bedrock.llm.model.stability.message import (
    parse_message,
    validate_last_message,
)
from aidial_adapter_bedrock.llm.model.stability.storage import save_to_storage
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages
from aidial_adapter_bedrock.utils.json import remove_nones
from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel
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


class StabilityV2Response(BaseModel):
    images: List[str] | None
    # None will indicate that the request was successful
    # Possible values:
    # "Filter reason: prompt"
    # "Filter reason: output image"
    # "Filter reason: input image"
    # "Inference error"
    # null
    finish_reasons: List[Optional[str]] | None

    def content(self) -> str:
        return " "

    def attachments(self) -> List[Attachment]:
        return [
            Attachment(
                title="Image",
                type="image/png",
                data=image,
            )
            for image in self.images or []
        ]

    def usage(self) -> TokenUsage:
        return TokenUsage(prompt_tokens=0, completion_tokens=1)

    def throw_if_error(self):
        error = next(filter(None, self.finish_reasons or []), None)
        if not error:
            return

        if error == "Inference error":
            raise InternalServerError(error)
        else:
            raise InvalidRequestError(code="content_filter", message=error)


AspectRatios = Literal[
    "16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"
]


# NOTE: The configuration is passed to the upstream endpoint *as is* a part of the request.
# Therefore, it's reasonable to allow extra fields to achieve forward-compatibility.
class StabilityImageConfiguration(ExtraAllowModel):
    aspect_ratio: AspectRatios | str | None = None
    negative_prompt: str | None = None


class StabilityV3Configuration(StabilityImageConfiguration):
    cfg_scale: float | None = None


Stability_V2_V3 = Literal[
    ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1,
    ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1_1,
    ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1,
    ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1_1,
    ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1,
    ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_5_LARGE_V1,
]


class Spec(BaseModel):
    image_to_image_supported: bool
    width_constraints: Tuple[int, int] | None
    height_constraints: Tuple[int, int] | None
    configuration_cls: Type[BaseModel]

    def validate_image(self, image: Resource) -> None:
        if self.width_constraints is None and self.height_constraints is None:
            return

        with Image.open(BytesIO(image.data)) as img:
            width, height = img.size

            for constraints, value, name in [
                (self.width_constraints, width, "width"),
                (self.height_constraints, height, "height"),
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


def _get_spec(deployment: Stability_V2_V3) -> Spec:
    match deployment:
        case (
            ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1
            | ChatCompletionDeployment.STABILITY_STABLE_IMAGE_CORE_V1_1
            | ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1
            | ChatCompletionDeployment.STABILITY_STABLE_IMAGE_ULTRA_V1_1
        ):
            return Spec(
                image_to_image_supported=False,
                width_constraints=None,
                height_constraints=None,
                configuration_cls=StabilityImageConfiguration,
            )
        case (
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_LARGE_V1
            | ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_3_5_LARGE_V1
        ):
            return Spec(
                image_to_image_supported=True,
                width_constraints=(640, 1536),
                height_constraints=(640, 1536),
                configuration_cls=StabilityV3Configuration,
            )
        case _:
            return assert_never(deployment)


class StabilityV2Adapter(ChatCompletionAdapter):
    deployment: AdapterDeployment[Stability_V2_V3]
    client: Bedrock
    storage: Optional[FileStorage]
    spec: Spec

    @classmethod
    def create(
        cls,
        client: Bedrock,
        deployment: AdapterDeployment[Stability_V2_V3],
        api_key: str,
    ):
        storage: Optional[FileStorage] = create_file_storage(api_key)
        return cls(
            client=client,
            deployment=deployment,
            storage=storage,
            spec=_get_spec(deployment.reference_deployment_id),
        )

    async def configuration(self) -> Type[BaseModel]:
        return self.spec.configuration_cls

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        validate_last_message(messages)
        return list(range(len(messages) - 1))

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:

        configuration = params.parse_configuration(await self.configuration())
        configuration_dict = (
            {} if configuration is None else configuration.dict()
        )

        message = validate_last_message(messages)
        text_prompt, image_resources = parse_message(
            message, SUPPORTED_IMAGE_TYPES
        )

        if not self.spec.image_to_image_supported and image_resources:
            raise UserError("Image-to-Image is not supported")
        if len(image_resources) > 1:
            raise UserError("Only one input image is supported")

        if self.spec.image_to_image_supported and image_resources:
            image_resource = await _download_resource(
                image_resources[0], self.storage
            )
            self.spec.validate_image(image_resource)
        else:
            image_resource = None

        if not text_prompt:
            raise UserError("Text prompt is required")

        response, _ = await self.client.ainvoke_non_streaming(
            self.deployment.upstream_deployment_id,
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
                    "seed": params.seed,
                    **configuration_dict,
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
