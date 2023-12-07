from typing import Callable, List, Optional, Tuple, TypedDict

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)


class RolePrefixes(TypedDict):
    system: Optional[str]
    human: Optional[str]
    ai: Optional[str]


class ChatEmulator(BaseModel):
    prelude_template: Optional[str]
    add_role_prefix: Callable[[BaseMessage, int], bool]
    add_invitation: bool
    fallback_to_completion: bool
    role_prefixes: RolePrefixes
    separator: str

    @property
    def _prelude(self) -> Optional[str]:
        if self.prelude_template is None:
            return None
        return self.prelude_template.format(**self.role_prefixes)

    def _get_role(self, message: BaseMessage) -> Optional[str]:
        if isinstance(message, HumanMessage):
            return self.role_prefixes["human"]
        elif isinstance(message, AIMessage):
            return self.role_prefixes["ai"]
        elif isinstance(message, BaseMessage):
            return self.role_prefixes["system"]
        else:
            raise ValueError(f"Unknown message type: {message.type}")

    def _format_message(self, message: BaseMessage, idx: int) -> str:
        role = self._get_role(message)

        if role is None or not self.add_role_prefix(message, idx):
            role_prefix = ""
        else:
            role_prefix = role + " "

        separator = "" if idx == 0 else self.separator

        return (separator + role_prefix + message.content.lstrip()).rstrip()

    def display(self, messages: List[BaseMessage]) -> Tuple[str, List[str]]:
        if (
            len(messages) == 1
            and isinstance(messages[0], HumanMessage)
            and self.fallback_to_completion
        ):
            return messages[0].content, []

        ret: List[str] = []

        if self._prelude is not None:
            ret.append(self._prelude)

        for message in messages:
            ret.append(self._format_message(message, len(ret)))

        if self.add_invitation:
            ret.append(self._format_message(AIMessage(content=""), len(ret)))

        stop_sequences: List[str] = []
        human_role = self.role_prefixes["human"]
        if human_role is not None:
            stop_sequences = [self.separator + human_role]

        return "".join(ret), stop_sequences


default_conf = ChatEmulator(
    prelude_template="""
You are a helpful assistant participating in a dialog with a user.
The messages from the user start with "{ai}".
The messages from you start with "{human}".
Reply to the last message from the user taking into account the preceding dialog history.
====================
""".strip(),
    add_role_prefix=lambda *_: True,
    add_invitation=True,
    fallback_to_completion=True,
    role_prefixes=RolePrefixes(
        system="Human:",
        human="Human:",
        ai="Assistant:",
    ),
    separator="\n\n",
)
