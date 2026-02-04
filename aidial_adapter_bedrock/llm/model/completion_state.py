from typing import Dict

from aidial_adapter_anthropic.dial.token_usage import TokenUsage
from aidial_sdk.chat_completion import FinishReason
from pydantic import BaseModel

FinishReasons = Dict[int, FinishReason]


class CompletionState(BaseModel):
    finish_reasons: Dict[int, FinishReason] = {}
    usage: TokenUsage = TokenUsage()

    def get_single_finish_reason(self) -> FinishReason | None:
        return next((r for r in self.finish_reasons.values()), None)
