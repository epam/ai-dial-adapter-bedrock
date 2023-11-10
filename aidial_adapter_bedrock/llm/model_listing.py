from typing import List, TypedDict

import boto3


class BedrockModelId(TypedDict):
    modelArn: str
    modelId: str


def get_bedrock_models(region: str) -> List[BedrockModelId]:
    session = boto3.Session()
    bedrock = session.client("bedrock", region)
    return bedrock.list_foundation_models()["modelSummaries"]
