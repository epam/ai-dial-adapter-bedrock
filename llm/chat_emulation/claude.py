from enum import Enum
from typing import List

from langchain.schema import BaseMessage
from pydantic import BaseModel

from llm.exceptions import ValidationError


class ClaudeRole(Enum):
    HUMAN = "Human"
    ASSISTANT = "Assistant"


class ClaudeMessage(BaseModel):
    role: ClaudeRole
    content: str


class ClaudeHistory:
    history: List[ClaudeMessage]

    def __init__(self):
        self.history = []

    def add(self, msg: ClaudeMessage):
        if len(self.history) > 0 and self.history[-1].role == msg.role:
            self.history[-1].content += " " + msg.content
        else:
            self.history.append(msg)

    def print(self) -> str:
        return "".join(
            [
                f"\n\n{msg.role.value}: {msg.content.lstrip()}".rstrip()
                for msg in self.history
            ]
        )


def emulate(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise ValidationError("List of messages must not be empty")

    history = ClaudeHistory()
    for msg in prompt:
        role = (
            ClaudeRole.HUMAN
            if msg.type in ["system", "human"]
            else ClaudeRole.ASSISTANT
        )
        history.add(ClaudeMessage(role=role, content=msg.content))
    history.add(ClaudeMessage(role=ClaudeRole.ASSISTANT, content=""))

    return history.print()
