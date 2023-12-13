from typing import List, TypedDict

import boto3


class BedrockModelId(TypedDict):
    modelArn: str
    modelId: str


def get_all_bedrock_models(region: str) -> List[str]:
    session = boto3.Session()
    bedrock = session.client("bedrock", region)
    models: List[BedrockModelId] = bedrock.list_foundation_models()[
        "modelSummaries"
    ]
    return [model["modelId"] for model in models]
