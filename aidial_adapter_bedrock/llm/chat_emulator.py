from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, TypedDict

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    BaseMessage,
    HumanRegularMessage,
)


class ChatEmulator(ABC, BaseModel):
    @abstractmethod
    def display(self, messages: List[BaseMessage]) -> Tuple[str, List[str]]:
        """Returns a prompt string and a list of stop sequences."""

    @abstractmethod
    def get_ai_cue(self) -> Optional[str]:
        pass


class CueMapping(TypedDict):
    system: Optional[str]
    human: Optional[str]
    ai: Optional[str]


class BasicChatEmulator(ChatEmulator):
    prelude_template: Optional[str]
    add_cue: Callable[[BaseMessage, int], bool]
    add_invitation_cue: bool
    fallback_to_completion: bool
    cues: CueMapping
    separator: str

    @property
    def _prelude(self) -> Optional[str]:
        if self.prelude_template is None:
            return None
        return self.prelude_template.format(**self.cues)

    def _get_cue(self, message: BaseMessage) -> Optional[str]:
        if isinstance(message, HumanRegularMessage):
            return self.cues["human"]
        elif isinstance(message, AIRegularMessage):
            return self.cues["ai"]
        elif isinstance(message, BaseMessage):
            return self.cues["system"]
        else:
            raise ValueError(f"Unknown message type: {message.type}")

    def _format_message(self, message: BaseMessage, idx: int) -> str:
        cue = self._get_cue(message)

        if cue is None or not self.add_cue(message, idx):
            cue_prefix = ""
        else:
            cue_prefix = cue + " "

        return (cue_prefix + message.text_content.lstrip()).rstrip()

    def get_ai_cue(self) -> Optional[str]:
        return self.cues["ai"]

    def display(self, messages: List[BaseMessage]) -> Tuple[str, List[str]]:
        if (
            self.fallback_to_completion
            and len(messages) == 1
            and isinstance(messages[0], HumanRegularMessage)
        ):
            return messages[0].text_content, []

        ret: List[str] = []

        if self._prelude is not None:
            ret.append(self._prelude)

        for message in messages:
            ret.append(self._format_message(message, len(ret)))

        if self.add_invitation_cue:
            ret.append(
                self._format_message(AIRegularMessage(content=""), len(ret))
            )

        stop_sequences: List[str] = []
        human_role = self.cues["human"]
        if human_role is not None:
            stop_sequences = [self.separator + human_role]

        return self.separator.join(ret), stop_sequences


default_emulator = BasicChatEmulator(
    prelude_template="""
You are a helpful assistant participating in a dialog with a user.
The messages from the user start with "{ai}".
The messages from you start with "{human}".
Reply to the last message from the user taking into account the preceding dialog history.
====================
""".strip(),
    add_cue=lambda *_: True,
    add_invitation_cue=True,
    fallback_to_completion=True,
    cues=CueMapping(
        system="Human:",
        human="Human:",
        ai="Assistant:",
    ),
    separator="\n\n",
)
