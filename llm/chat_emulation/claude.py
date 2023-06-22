from typing import List

from langchain.schema import BaseMessage


def emulate(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise Exception("Prompt must not be empty")

    msgs: List[str] = []
    for msg in prompt:
        role = "Human" if msg.type in ["system", "human"] else "Assistant"
        msgs.append(f"{role}: {msg.content}")
    msgs.append("Assistant:")

    return "".join([f"\n\n{msg}" for msg in msgs])
