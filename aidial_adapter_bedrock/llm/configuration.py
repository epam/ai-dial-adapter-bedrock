from typing import Literal

from aidial_sdk.chat_completion import Request as ChatCompletionRequest
from pydantic import Field

from aidial_adapter_bedrock.utils.pydantic import ExtraForbidModel


class BedrockPerformanceConfig(ExtraForbidModel):
    latency: Literal["standard", "optimized"] | str | None = Field(
        default=None,
        description="Latency configuration",
    )


class BaseBedrockConfiguration(ExtraForbidModel):
    performanceConfig: BedrockPerformanceConfig | None = Field(
        default=None,
        description="Bedrock performance configuration",
    )


def has_performance_configuration(request: ChatCompletionRequest) -> bool:
    configuration = cf.configuration if (cf := request.custom_fields) else None
    return configuration is not None and "performanceConfig" in configuration
