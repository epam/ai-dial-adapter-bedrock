from typing import List, Optional

from langchain.schema import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from chat_client.init import BedrockModels


class Message(BaseModel):
    role: str
    content: Optional[str]
    name: Optional[str]
    function_call: Optional[str]

    class Config:
        extra = "allow"

    def to_base_message(self) -> BaseMessage:
        assert self.content is not None
        match self.role:
            case "system":
                return SystemMessage(content=self.content)
            case "user":
                return HumanMessage(content=self.content)
            case "assistant":
                return AIMessage(content=self.content)
            case _:
                raise ValueError(f"Unknown role: {self.role}")


# Direct translation of https://platform.openai.com/docs/api-reference/chat/create
class ChatCompletionQuery(BaseModel):
    model: BedrockModels
    messages: List[Message]
    max_tokens: Optional[int]

    class Config:
        extra = "allow"
