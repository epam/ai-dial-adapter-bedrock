from typing import List, Optional

from aidial_sdk.chat_completion import Request
from pydantic import BaseModel

from aidial_adapter_bedrock.llm.tools_emulation.base import ToolConfig


class ModelParameters(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: List[str] = []
    max_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    stream: bool = False
    tool_config: Optional[ToolConfig] = None

    @classmethod
    def create(cls, request: Request) -> "ModelParameters":
        stop: List[str] = []
        if request.stop is not None:
            stop = (
                [request.stop]
                if isinstance(request.stop, str)
                else request.stop
            )

        return cls(
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=stop,
            max_tokens=request.max_tokens,
            max_prompt_tokens=request.max_prompt_tokens,
            stream=request.stream,
            tool_config=ToolConfig.from_request(request),
        )

    def add_stop_sequences(self, stop: List[str]) -> "ModelParameters":
        return self.copy(update={"stop": [*self.stop, *stop]})
