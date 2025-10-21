from typing import List, Literal

from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import BetaThinkingConfigParam as ThinkingConfigParam
from pydantic import Field

from aidial_adapter_bedrock.utils.pydantic import ExtraForbidModel


class ThinkingConfigEnabled(ExtraForbidModel):
    type: Literal["enabled"]
    budget_tokens: int

    def to_claude(self) -> ThinkingConfigParam:
        return {"type": "enabled", "budget_tokens": self.budget_tokens}


class ThinkingConfigDisabled(ExtraForbidModel):
    type: Literal["disabled"]

    def to_claude(self) -> ThinkingConfigParam:
        return {"type": "disabled"}


class ClaudeConfiguration(ExtraForbidModel):
    betas: List[AnthropicBetaParam] | None = Field(
        default=None,
        description="List of beta features to enable. Make sure to check if the given feature is supported by the Claude deployment you are using.",
    )
    enable_citations: bool = False


class ClaudeConfigurationWithThinking(ClaudeConfiguration):
    # NOTE: once migrated to Pydantic v2 we can use TypeAdapter over
    # the anthropic's ThinkingConfigParam class directly.
    thinking: ThinkingConfigEnabled | ThinkingConfigDisabled | None = None


Configuration = ClaudeConfiguration | ClaudeConfigurationWithThinking
