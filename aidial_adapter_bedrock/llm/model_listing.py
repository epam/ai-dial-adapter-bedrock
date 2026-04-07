import boto3
from typing_extensions import TypedDict


class BedrockModelId(TypedDict):
    modelArn: str
    modelId: str


def get_all_bedrock_models(region: str) -> list[str]:
    session = boto3.Session()
    bedrock = session.client("bedrock", region)
    models: list[BedrockModelId] = bedrock.list_foundation_models()[
        "modelSummaries"
    ]
    return [model["modelId"] for model in models]
    return [model["modelId"] for model in models]
