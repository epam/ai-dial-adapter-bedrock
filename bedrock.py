import json
from typing import List, TypedDict

import boto3

from utils.env import get_env


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
    def __init__(self, region: str = "us-east-1"):
        session = boto3.Session(
            aws_access_key_id=get_env("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_env("AWS_SECRET_ACCESS_KEY"),
        )

        self.bedrock = session.client(
            "bedrock", region, endpoint_url=f"https://bedrock.{region}.amazonaws.com"
        )

    def available_models(self) -> List[BedrockModelId]:
        return self.bedrock.list_foundation_models()["modelSummaries"]

    def predict(self, modelId: str, prompt: str) -> BedrockResponse:
        response = self.bedrock.invoke_model(
            body=json.dumps({"inputText": prompt}),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )

        return json.loads(response["body"].read())
