from typing import List, Mapping, Optional, Union

from aidial_sdk.chat_completion import Request
from pydantic import BaseModel


class ModelParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Mapping[int, float]] = None
    stream: bool = False

    @classmethod
    def create(cls, request: Request) -> "ModelParameters":
        return cls(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
            max_tokens=request.max_tokens,
            max_prompt_tokens=request.max_prompt_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            stream=request.stream,
        )

    def add_stop_sequences(self, stop: List[str]) -> "ModelParameters":
        if len(stop) == 0:
            return self

        self_stop: List[str] = []
        if self.stop is not None:
            if isinstance(self.stop, str):
                self_stop = [self.stop]
            else:
                self_stop = self.stop

        return self.copy(update={"stop": [*self_stop, *stop]})
