from typing import Literal

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
