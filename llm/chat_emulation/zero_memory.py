from typing import List

from langchain.schema import BaseMessage


def emulate(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise Exception("Prompt must not be empty")
    return prompt[-1].content
