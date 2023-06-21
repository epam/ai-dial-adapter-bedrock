import json
from typing import List, TypedDict

import boto3
from langchain.schema import BaseMessage

from llm.chat_emulation import ChatEmulationType, history_compression


class BedrockModelId(TypedDict):
    modelArn: str
    modelId: str


class BedrockResult(TypedDict):
    tokenCount: int
    outputText: str


class BedrockResponse(TypedDict):
    inputTextTokenCount: int
    results: List[BedrockResult]


class BedrockModel:
    def __init__(
        self, model_id: str, chat_emulation_type: ChatEmulationType, region: str
    ):
        session = boto3.Session()

        self.bedrock = session.client(
            "bedrock",
            region,
            endpoint_url=f"https://bedrock.{region}.amazonaws.com",
        )
        self.model_id = model_id
        self.chat_emulation_type = chat_emulation_type

    def available_models(self) -> List[BedrockModelId]:
        return self.bedrock.list_foundation_models()["modelSummaries"]

    def predict(self, prompt: str) -> BedrockResponse:
        response = self.bedrock.invoke_model(
            body=json.dumps({"inputText": prompt}),
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
        )

        return json.loads(response["body"].read())

    def chat(self, prompt: List[BaseMessage]) -> BedrockResponse:
        return self.predict(
            history_compression(self.chat_emulation_type, prompt)
        )
