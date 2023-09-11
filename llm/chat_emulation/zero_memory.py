from typing import List

from langchain.schema import BaseMessage

from llm.exceptions import ValidationError


def emulate(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise ValidationError("Prompt must not be empty")
    return prompt[-1].content
