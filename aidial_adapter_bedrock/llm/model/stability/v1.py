from enum import Enum
from typing import Any, Dict, List, Optional

from aidial_sdk.chat_completion import Attachment, Message
from pydantic import BaseModel, Field

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import (
    ChatCompletionAdapter,
    default_preprocess_messages,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.decorator.tools_emulator import (
    tools_emulator_decorator,
)
from aidial_adapter_bedrock.llm.errors import UserError, ValidationError
from aidial_adapter_bedrock.llm.model.stability.message import (
    parse_message,
    validate_last_message,
)
from aidial_adapter_bedrock.llm.model.stability.storage import save_to_storage
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)
from aidial_adapter_bedrock.llm.truncate_prompt import DiscardedMessages


class StabilityStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class StabilityError(BaseModel):
    id: str
    message: str
    name: str


class StabilityArtifact(BaseModel):
    seed: int
    base64: str
    finish_reason: str = Field(alias="finishReason")


class StabilityResponse(BaseModel):
    # TODO: Use tagged union artifacts/error
    result: StabilityStatus
    artifacts: Optional[list[StabilityArtifact]]
    error: Optional[StabilityError]

    def content(self) -> str:
        self._throw_if_error()
        # NOTE: text-to-text models aren't expected to generate empty strings.
        # So since we represent text-to-image model (Stability) as
        # a text-to-text model (via chat completion interface),
        # we need to return something.
        return " "

    def attachments(self) -> list[Attachment]:
        self._throw_if_error()
        return [
            Attachment(
                title="Image",
                type="image/png",
                data=self.artifacts[0].base64,  # type: ignore
            )
        ]

    def usage(self) -> TokenUsage:
        self._throw_if_error()
        return TokenUsage(
            prompt_tokens=0,
            completion_tokens=1,
        )

    def _throw_if_error(self):
        if self.result == StabilityStatus.ERROR:
            raise Exception(self.error.message)  # type: ignore


def create_request(prompt: str) -> Dict[str, Any]:
    return {"text_prompts": [{"text": prompt}]}


def create_adapter(
    client: Bedrock, model: str, api_key: str
) -> ChatCompletionAdapter:
    return compose_decorators(
        preprocess_messages_decorator(default_preprocess_messages),
        replicator_decorator(),
        tools_emulator_decorator(default_tools_emulator),
    )(StabilityV1Adapter.create(client, model, api_key))


class StabilityV1Adapter(ChatCompletionAdapter):
    model: str
    client: Bedrock
    storage: Optional[FileStorage]

    @classmethod
    def create(cls, client: Bedrock, model: str, api_key: str):
        storage: Optional[FileStorage] = create_file_storage(api_key)
        return cls(client=client, model=model, storage=storage)

    async def chat(
        self,
        consumer: Consumer,
        params: ModelParameters,
        messages: List[Message],
    ) -> None:

        message = validate_last_message(messages)
        text_prompt, image_resources = parse_message(message, [])

        if image_resources:
            raise UserError("Image-to-Image is not supported")

        if text_prompt is None:
            raise ValidationError("Content of the last message is missing")

        args = create_request(text_prompt)
        response, _headers = await self.client.ainvoke_non_streaming(
            self.model, args
        )

        resp = StabilityResponse.parse_obj(response)
        consumer.append_content(resp.content())
        consumer.close_content()

        consumer.add_usage(resp.usage())

        for attachment in resp.attachments():
            if self.storage:
                attachment = await save_to_storage(self.storage, attachment)
            consumer.add_attachment(attachment)

    async def compute_discarded_messages(
        self, params: ModelParameters, messages: List[Message]
    ) -> DiscardedMessages | None:
        validate_last_message(messages)
        return list(range(len(messages) - 1))
