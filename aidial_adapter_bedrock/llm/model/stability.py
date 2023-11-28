import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation.zero_memory_chat import (
    ZeroMemoryChatHistory,
)
from aidial_adapter_bedrock.llm.chat_model import ChatModel, ChatPrompt
from aidial_adapter_bedrock.llm.consumer import Attachment, Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.utils.concurrency import make_async


class ResponseData(BaseModel):
    mime_type: str
    name: str
    content: str


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
    result: str
    artifacts: Optional[list[StabilityArtifact]]
    error: Optional[StabilityError]

    def content(self) -> str:
        self._throw_if_error()
        return ""

    def data(self) -> list[ResponseData]:
        self._throw_if_error()
        return [
            ResponseData(
                mime_type="image/png",
                name="image",
                content=self.artifacts[0].base64,  # type: ignore
            )
        ]

    def usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=0,
            completion_tokens=1,
        )

    def _throw_if_error(self):
        if self.result == StabilityStatus.ERROR:
            raise Exception(self.error.message)  # type: ignore


def prepare_input(prompt: str) -> Dict[str, Any]:
    return {"text_prompts": [{"text": prompt}]}


class StabilityAdapter(ChatModel):
    def __init__(
        self,
        bedrock: Any,
        model_id: str,
    ):
        super().__init__(model_id)
        self.bedrock = bedrock

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
        return await make_async(
            lambda args: self._call(*args), (consumer, prompt)
        )

    def _call(self, consumer: Consumer, prompt: str):
        model_response = self.bedrock.invoke_model(
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(prepare_input(prompt)),
        )

        body = json.loads(model_response["body"].read())
        resp = StabilityResponse.parse_obj(body)

        consumer.append_content(resp.content())
        consumer.add_usage(resp.usage())

        for data in resp.data():
            consumer.add_attachment(
                Attachment(
                    title=data.name,
                    data=data.content,
                    type=data.mime_type,
                )
            )
