from enum import Enum
from typing import List

from langchain.schema import BaseMessage


class ChatEmulationType(Enum):
    ZERO_MEMORY = "zero_memory"
    META_CHAT = "meta_chat"


def zero_memory_compression(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise Exception("Prompt must not be empty")
    return prompt[-1].content


meta_chat_stop = "[HUMAN]"

meta_chat_prelude = """
You are participating in a dialog with user.
The messages from user are prefixed with "[HUMAN]".
The messages from you are prefixed with "[AI]".
The messages providing additional user instructions are prefixed with "[SYSTEM]".
Reply to the last message from user taking into account the preceding dialog history.
====================
""".strip()


def meta_chat_compression(prompt: List[BaseMessage]) -> str:
    if len(prompt) == 0:
        raise Exception("Prompt must not be empty")

    msgs = [meta_chat_prelude]
    for msg in prompt:
        msgs.append(f"[{msg.type.upper()}] {msg.content}")
    msgs.append("[AI] ")

    return "\n".join(msgs)


def history_compression(
    emulation_type: ChatEmulationType, prompt: List[BaseMessage]
) -> str:
    match emulation_type:
        case ChatEmulationType.ZERO_MEMORY:
            return zero_memory_compression(prompt)
        case ChatEmulationType.META_CHAT:
            return meta_chat_compression(prompt)
        case _:
            raise Exception(f"Invalid emulation type: {emulation_type}")
