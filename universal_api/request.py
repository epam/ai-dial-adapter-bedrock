from typing import List, Literal, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: Optional[str]
    name: Optional[str] = None
    function_call: Optional[str] = None

    class Config:
        extra = "allow"


class ChatCompletionParameters(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None


# Direct translation of https://platform.openai.com/docs/api-reference/chat/create
class ChatCompletionRequest(ChatCompletionParameters, BaseModel):
    messages: List[Message]

    class Config:
        extra = "allow"
