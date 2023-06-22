from typing import List

from langchain.schema import BaseMessage

stop = "[HUMAN]"

prelude = """
You are participating in a dialog with user.
The messages from user are prefixed with "[HUMAN]".
The messages from you are prefixed with "[AI]".
The messages providing additional user instructions are prefixed with "[SYSTEM]".
Reply to the last message from user taking into account the preceding dialog history.
====================
""".strip()


def emulate(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise Exception("Prompt must not be empty")

    msgs = [prelude]
    for msg in prompt:
        msgs.append(f"[{msg.type.upper()}] {msg.content}")
    msgs.append("[AI] ")

    return "\n".join(msgs)
