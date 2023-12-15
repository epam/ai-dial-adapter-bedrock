import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.auth import Auth
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    upload_base64_file,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import ChatModel, ChatPrompt
from aidial_adapter_bedrock.llm.consumer import Attachment, Consumer
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.utils.env import get_env


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
        return ""

    def attachments(self) -> list[Attachment]:
        self._throw_if_error()
        return [
            Attachment(
                title="image",
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


async def save_to_storage(
    storage: FileStorage, attachment: Attachment
) -> Attachment:
    if (
        attachment.type is not None
        and attachment.type.startswith("image/")
        and attachment.data is not None
    ):
        response = await upload_base64_file(
            storage, attachment.data, attachment.type
        )
        return Attachment(
            title=attachment.title,
            type=attachment.type,
            url=response["path"] + "/" + response["name"],
        )

    return attachment


DIAL_USE_FILE_STORAGE = (
    os.getenv("DIAL_USE_FILE_STORAGE", "false").lower() == "true"
)

if DIAL_USE_FILE_STORAGE:
    DIAL_URL = get_env(
        "DIAL_URL", "DIAL_URL must be set to use the DIAL file storage"
    )


class StabilityAdapter(ChatModel):
    client: Bedrock
    storage: Optional[FileStorage]

    def __init__(
        self, client: Bedrock, model: str, get_auth: Callable[[], Auth]
    ):
        super().__init__(model)
        self.client = client
        self.storage = None

        if DIAL_USE_FILE_STORAGE:
            self.storage = FileStorage(
                dial_url=DIAL_URL,
                auth=get_auth(),
                base_dir="images/stable-diffusion",
            )

    def _prepare_prompt(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> ChatPrompt:
        if len(messages) == 0:
            raise ValidationError("List of messages must not be empty")

        return ChatPrompt(
            text=messages[-1].content,
            stop_sequences=[],
            discarded_messages=len(messages) - 1,
        )

    async def _apredict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt)
        response = await self.client.ainvoke_non_streaming(self.model, args)

        resp = StabilityResponse.parse_obj(response)
        consumer.append_content(resp.content())
        consumer.add_usage(resp.usage())

        for attachment in resp.attachments():
            if self.storage:
                attachment = await save_to_storage(self.storage, attachment)
            consumer.add_attachment(attachment)
