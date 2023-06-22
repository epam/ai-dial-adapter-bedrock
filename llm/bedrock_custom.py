import json
from typing import List, TypedDict

import boto3
from langchain.schema import BaseMessage

from llm.chat_emulation import emulate_chat
from llm.chat_emulation.types import ChatEmulationType
from utils.printing import print_info


class BedrockModelId(TypedDict):
    modelArn: str
    modelId: str


class BedrockResult(TypedDict):
    tokenCount: int
    outputText: str


class BedrockResponse(TypedDict):
    inputTextTokenCount: int
    results: List[BedrockResult]


class BedrockModels:
    def __init__(self, region: str = "us-east-1"):
        session = boto3.Session()

        self.bedrock = session.client(
            "bedrock",
            region,
            endpoint_url=f"https://bedrock.{region}.amazonaws.com",
        )

    def models(self) -> List[BedrockModelId]:
        return self.bedrock.list_foundation_models()["modelSummaries"]


class BedrockModel:
    def __init__(
        self,
        model_id: str,
        chat_emulation_type: ChatEmulationType,
        region: str = "us-east-1",
    ):
        session = boto3.Session()

        self.bedrock = session.client(
            "bedrock",
            region,
            endpoint_url=f"https://bedrock.{region}.amazonaws.com",
        )
        self.model_id = model_id
        self.chat_emulation_type = chat_emulation_type

    def predict(self, prompt: str) -> BedrockResponse:
        response = self.bedrock.invoke_model(
            body=json.dumps({"inputText": prompt}),
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
        )

        return json.loads(response["body"].read())

    def chat(self, prompt: List[BaseMessage]) -> BedrockResponse:
        prompt1 = emulate_chat(self.model_id, self.chat_emulation_type, prompt)
        print_info(prompt1)
        return self.predict(prompt1)
