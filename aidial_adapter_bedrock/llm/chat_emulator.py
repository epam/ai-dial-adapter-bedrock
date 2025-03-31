from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, List, Optional, Tuple, TypedDict

from aidial_sdk.chat_completion import Message, Role
from pydantic import BaseModel

import aidial_adapter_bedrock.utils.stream as stream_utils
from aidial_adapter_bedrock.dial_api.request import (
    ModelParameters,
    collect_text_content,
)


class ChatEmulator(ABC, BaseModel):
    @abstractmethod
    def display(self, messages: List[Message]) -> Tuple[str, List[str]]:
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
    should_prefix_with_cue: Callable[[Message, int], bool]
    should_add_invitation_cue: bool
    should_fallback_to_completion: bool
    cues: CueMapping
    separator: str

    @property
    def _prelude(self) -> Optional[str]:
        if self.prelude_template is None:
            return None
        return self.prelude_template.format(**self.cues)

    def _get_cue(self, message: Message) -> Optional[str]:
        match message.role:
            case Role.USER:
                return self.cues["human"]
            case Role.ASSISTANT:
                return self.cues["ai"]
            case Role.SYSTEM | Role.DEVELOPER:
                return self.cues["system"]
            case _:
                raise ValueError(f"Unexpected message type: {message.role}")

    def _format_message(self, message: Message, idx: int) -> str:
        cue = self._get_cue(message)

        if cue is None or not self.should_prefix_with_cue(message, idx):
            cue_prefix = ""
        else:
            cue_prefix = cue + " "

        return (
            cue_prefix + collect_text_content(message.content).lstrip()
        ).rstrip()

    def get_ai_cue(self) -> Optional[str]:
        return self.cues["ai"]

    def display(self, messages: List[Message]) -> Tuple[str, List[str]]:
        if (
            self.should_fallback_to_completion
            and len(messages) == 1
            and messages[0].role == Role.USER
        ):
            return collect_text_content(messages[0].content), []

        ret: List[str] = []

        if self._prelude is not None:
            ret.append(self._prelude)

        for message in messages:
            ret.append(self._format_message(message, len(ret)))

        if self.should_add_invitation_cue:
            ret.append(
                self._format_message(
                    Message(role=Role.ASSISTANT, content=""), len(ret)
                )
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
    should_prefix_with_cue=lambda *_: True,
    should_add_invitation_cue=True,
    should_fallback_to_completion=True,
    cues=CueMapping(
        system="Human:",
        human="Human:",
        ai="Assistant:",
    ),
    separator="\n\n",
)


def post_process_completion_stream(
    params: ModelParameters,
    emulator: ChatEmulator,
    stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    # Removing leading spaces
    stream = stream_utils.lstrip(stream)

    # Model may occasionally start responding with its cue.
    ai_cue = emulator.get_ai_cue()
    if ai_cue is not None:
        stream = stream_utils.remove_prefix(stream, ai_cue)
        stream = stream_utils.lstrip(stream)

    # The model may not support stop sequences, so do it manually
    if params.stop:
        stream = stream_utils.stop_at(stream, params.stop)

    # After all the post processing, the stream may become empty.
    # To avoid this, add a space to the stream.
    stream = stream_utils.ensure_not_empty(stream, " ")

    return stream
