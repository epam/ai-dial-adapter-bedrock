from typing import List, Tuple

from langchain.schema import AIMessage, BaseMessage

from utils.operators import Unary
from utils.text import enforce_stop_tokens, remove_prefix

HUMAN = "Human"
ASSISTANT = "Assistant"
SYSTEM = "System"

prelude = f"""
You are a helpful assistant participating in a dialog with a user.
The messages from the user start with "{HUMAN}:".
The messages from you start with "{ASSISTANT}:".
Reply to the last message from the user taking into account the preceding dialog history.
====================
""".strip()


def type_to_role(ty: str) -> str:
    roles = {"human": HUMAN, "system": HUMAN, "ai": ASSISTANT}
    return roles.get(ty, ty)


def emulate(prompt: List[BaseMessage]) -> Tuple[str, Unary[str]]:
    if len(prompt) == 0:
        raise Exception("Prompt must not be empty")

    history = prompt.copy()
    history.append(AIMessage(content=""))

    msgs = [prelude]
    for msg in history:
        role = type_to_role(msg.type)
        msgs.append(f"\n\n{role}: {msg.content.lstrip()}".rstrip())

    return "".join(msgs), post_process


stop = f"{HUMAN}:"


def post_process(response: str) -> str:
    response = enforce_stop_tokens(response, [stop])
    response = remove_prefix(response.strip(), f"{ASSISTANT}:")
    return response.strip()
