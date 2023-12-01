import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.storage import (
    FileStorage,
    upload_base64_file,
)
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation.zero_memory_chat import (
    ZeroMemoryChatHistory,
)
from aidial_adapter_bedrock.llm.chat_model import ChatModel, ChatPrompt
from aidial_adapter_bedrock.llm.consumer import Attachment, Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.utils.concurrency import make_async
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


def prepare_input(prompt: str) -> Dict[str, Any]:
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


USE_DIAL_FILE_STORAGE = (
    os.getenv("USE_DIAL_FILE_STORAGE", "false").lower() == "true"
)

if USE_DIAL_FILE_STORAGE:
    DIAL_URL = get_env("DIAL_URL")
    DIAL_BEDROCK_API_KEY = get_env("DIAL_BEDROCK_API_KEY")


class StabilityAdapter(ChatModel):
    bedrock: Any
    storage: Optional[FileStorage]

    def __init__(self, bedrock: Any, model_id: str):
        super().__init__(model_id)
        self.bedrock = bedrock
        self.storage = None

        if USE_DIAL_FILE_STORAGE:
            self.storage = FileStorage(
                dial_url=DIAL_URL,
                api_key=DIAL_BEDROCK_API_KEY,
                base_dir="stability",
            )

    def _prepare_prompt(
        self, messages: List[BaseMessage], max_prompt_tokens: Optional[int]
    ) -> ChatPrompt:
        history = ZeroMemoryChatHistory.create(messages)
        return ChatPrompt(
            text=history.format(),
            stop_sequences=[],
            discarded_messages=history.discarded_messages,
        )

    async def _apredict(
        self, consumer: Consumer, model_params: ModelParameters, prompt: str
    ):
        model_response = await make_async(
            lambda args: self.bedrock.invoke_model(
                accept="application/json",
                contentType="application/json",
                modelId=args[0],
                body=args[1],
            ),
            (self.model_id, json.dumps(prepare_input(prompt))),
        )

        body = json.loads(model_response["body"].read())
        resp = StabilityResponse.parse_obj(body)

        consumer.append_content(resp.content())
        consumer.add_usage(resp.usage())

        for attachment in resp.attachments():
            if self.storage is not None:
                attachment = await save_to_storage(self.storage, attachment)
            consumer.add_attachment(attachment)
