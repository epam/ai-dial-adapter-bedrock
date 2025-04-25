from typing import Literal

from pydantic import BaseModel, Field


class BedrockPerformanceConfig(BaseModel):
    latency: Literal["standard", "optimized"] | str | None = Field(
        default=None,
        description="Latency configuration",
    )


class BaseBedrockConfiguration(BaseModel):
    performanceConfig: BedrockPerformanceConfig | None = Field(
        default=None,
        description="Bedrock performance configuration",
    )
