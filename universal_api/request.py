from typing import Mapping, Optional, Union

from aidial_sdk.chat_completion.request import (
    ChatCompletionRequest,
    N,
    Penalty,
    Stop,
    Temperature,
    TopP,
)
from pydantic import BaseModel, PositiveInt


class ModelParameters(BaseModel):
    temperature: Optional[Temperature] = None
    top_p: Optional[TopP] = None
    n: Optional[N] = None
    stop: Optional[Union[str, Stop]] = None
    max_tokens: Optional[PositiveInt] = None
    presence_penalty: Optional[Penalty] = None
    frequency_penalty: Optional[Penalty] = None
    logit_bias: Optional[Mapping[int, float]] = None

    @classmethod
    def create(cls, request: ChatCompletionRequest) -> "ModelParameters":
        return cls(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
        )
