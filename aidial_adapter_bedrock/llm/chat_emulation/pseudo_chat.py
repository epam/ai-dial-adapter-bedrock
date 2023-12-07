from typing import List, Optional, Tuple, TypedDict

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)


class RoleMapping(TypedDict):
    system: Optional[str]
    human: Optional[str]
    ai: Optional[str]


class PseudoChat(BaseModel):
    prelude_template: Optional[str]
    annotate_first: bool
    add_invitation: bool
    mapping: RoleMapping
    separator: str

    @property
    def _prelude(self) -> Optional[str]:
        if self.prelude_template is None:
            return None
        return self.prelude_template.format(**self.mapping)

    def _get_role(self, message: BaseMessage) -> Optional[str]:
        if isinstance(message, HumanMessage):
            return self.mapping["human"]
        elif isinstance(message, AIMessage):
            return self.mapping["ai"]
        elif isinstance(message, BaseMessage):
            return self.mapping["system"]
        else:
            raise ValueError(f"Unknown message type: {message.type}")

    def _format_message(self, message: BaseMessage, is_first: bool) -> str:
        role = self._get_role(message)

        if role is None:
            role_prefix = ""
        elif is_first and not self.annotate_first:
            role_prefix = ""
        else:
            role_prefix = role + " "

        separator = self.separator
        if is_first:
            separator = ""

        return (separator + role_prefix + message.content.lstrip()).rstrip()

    def display(self, messages: List[BaseMessage]) -> Tuple[str, List[str]]:
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            return messages[0].content, []

        ret: List[str] = []

        if self._prelude is not None:
            ret.append(self._prelude)

        for message in messages:
            ret.append(self._format_message(message, len(ret) == 0))

        if self.add_invitation:
            ret.append(
                self._format_message(AIMessage(content=""), len(ret) == 0)
            )

        stop_sequences: List[str] = []
        human_role = self.mapping["human"]
        if human_role is not None:
            stop_sequences = [self.separator + human_role]

        return "".join(ret), stop_sequences


noop_conf = PseudoChat(
    prelude_template=None,
    annotate_first=False,
    add_invitation=False,
    mapping=RoleMapping(system=None, human=None, ai=None),
    separator="",
)

default_conf = PseudoChat(
    prelude_template="""
You are a helpful assistant participating in a dialog with a user.
The messages from the user start with "{ai}".
The messages from you start with "{human}".
Reply to the last message from the user taking into account the preceding dialog history.
====================
""".strip(),
    annotate_first=True,
    add_invitation=True,
    mapping=RoleMapping(
        system="Human:",
        human="Human:",
        ai="Assistant:",
    ),
    separator="\n\n",
)
