from typing import List, Mapping, Optional, Union

from aidial_sdk.chat_completion import Request
from pydantic import BaseModel


class ModelParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Mapping[int, float]] = None

    @classmethod
    def create(cls, request: Request) -> "ModelParameters":
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
