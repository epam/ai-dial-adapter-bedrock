from typing import Literal, assert_never

from aidial_sdk.chat_completion import Request as ChatCompletionRequest
from aidial_sdk.deployment.tokenize import TokenizeRequest
from pydantic import Field

from aidial_adapter_bedrock.utils.pydantic import ExtraForbidModel


class BedrockPerformanceConfig(ExtraForbidModel):
    latency: Literal["standard", "optimized"] | str | None = Field(
        default=None, description="Latency configuration"
    )


class GuardrailConfig(ExtraForbidModel):
    guardrailIdentifier: str = Field(
        description="The identifier for the guardrail."
    )
    guardrailVersion: str = Field(description="The version of the guardrail.")
    trace: Literal["enabled", "disabled", "enabled_full"] | str | None = Field(
        default=None, description="The trace behavior for the guardrail."
    )
    streamProcessingMode: Literal["sync", "async"] | str | None = Field(
        default=None, description="The processing mode."
    )


class ConverseAPIConfiguration(ExtraForbidModel):
    performanceConfig: BedrockPerformanceConfig | None = Field(
        default=None, description="Model performance settings for the request."
    )
    guardrailConfig: GuardrailConfig | None = Field(
        default=None, description="Guardrail configuration"
    )


def has_converse_api_configuration(
    request: ChatCompletionRequest | TokenizeRequest,
) -> bool:
    match request:
        case ChatCompletionRequest():
            configuration = (
                cf.configuration if (cf := request.custom_fields) else None
            )
            return configuration is not None and (
                "performanceConfig" in configuration
                or "guardrailConfig" in configuration
            )
        case TokenizeRequest():
            for inp in request.inputs:
                if inp.type == "request":
                    configuration = (
                        cf.configuration
                        if (cf := inp.value.custom_fields)
                        else None
                    )
                    condition = configuration is not None and (
                        "performanceConfig" in configuration
                        or "guardrailConfig" in configuration
                    )
                    if not condition:
                        return False

            return True
        case _:
            assert_never(request)
