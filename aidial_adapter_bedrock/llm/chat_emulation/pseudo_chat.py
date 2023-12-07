from typing import Callable, List, Optional, Set, Tuple, TypedDict

from pydantic import BaseModel

from aidial_adapter_bedrock.llm.chat_emulation.history import (
    FormattedMessage,
    History,
    is_important_message,
)
from aidial_adapter_bedrock.llm.exceptions import ValidationError
from aidial_adapter_bedrock.llm.message import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from aidial_adapter_bedrock.utils.list import exclude_indices


class RoleMapping(TypedDict):
    system: Optional[str]
    human: Optional[str]
    ai: Optional[str]


class PseudoChatConf(BaseModel):
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


noop_conf = PseudoChatConf(
    prelude_template=None,
    annotate_first=False,
    add_invitation=False,
    mapping=RoleMapping(system=None, human=None, ai=None),
    separator="",
)

default_conf = PseudoChatConf(
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


class PseudoChatHistory(History):
    stop_sequences: List[str]
    pseudo_history_conf: PseudoChatConf

    def trim(
        self, count_tokens: Callable[[str], int], max_prompt_tokens: int
    ) -> Tuple["PseudoChatHistory", int]:
        message_tokens = [
            count_tokens(message.text) for message in self.messages
        ]
        prompt_tokens = sum(message_tokens)
        if prompt_tokens <= max_prompt_tokens:
            return self, 0

        discarded_messages: Set[int] = set()
        source_messages_count: int = 0
        last_source_message: Optional[BaseMessage] = None
        for index, message in enumerate(self.messages):
            if message.source_message:
                source_messages_count += 1
                last_source_message = message.source_message

            if message.is_important:
                continue

            discarded_messages.add(index)
            prompt_tokens -= message_tokens[index]
            if prompt_tokens <= max_prompt_tokens:
                return (
                    PseudoChatHistory.create(
                        messages=[
                            message.source_message
                            for message in exclude_indices(
                                self.messages, discarded_messages
                            )
                            if message.source_message
                        ],
                        conf=self.pseudo_history_conf,
                    ),
                    len(discarded_messages),
                )

        if discarded_messages:
            discarded_messages_count = len(discarded_messages)
            if (
                source_messages_count - discarded_messages_count == 1
                and isinstance(last_source_message, HumanMessage)
            ):
                history = PseudoChatHistory.create(
                    messages=[last_source_message],
                    conf=self.pseudo_history_conf,
                )
                prompt_tokens = sum(
                    count_tokens(message.text) for message in history.messages
                )
                if prompt_tokens <= max_prompt_tokens:
                    return history, discarded_messages_count

            raise ValidationError(
                f"The token size of system messages and the last user message ({prompt_tokens}) exceeds"
                f" prompt token limit ({max_prompt_tokens})."
            )

        raise ValidationError(
            f"Prompt token size ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})."
        )

    @classmethod
    def create(
        cls, messages: List[BaseMessage], conf: PseudoChatConf
    ) -> "PseudoChatHistory":
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            message = messages[0]
            return cls(
                messages=[
                    FormattedMessage(
                        text=message.content,
                        source_message=message,
                    )
                ],
                stop_sequences=[],
                pseudo_history_conf=conf,
            )

        formatted_messages: List[FormattedMessage] = []

        if conf._prelude is not None:
            formatted_messages.append(FormattedMessage(text=conf._prelude))

        for idx, message in enumerate(messages):
            formatted_messages.append(
                FormattedMessage(
                    text=conf._format_message(
                        message, len(formatted_messages) == 0
                    ),
                    source_message=message,
                    is_important=is_important_message(messages, idx),
                )
            )

        if conf.add_invitation:
            formatted_messages.append(
                FormattedMessage(
                    text=conf._format_message(
                        AIMessage(content=""), len(formatted_messages) == 0
                    )
                )
            )

        return cls(
            messages=formatted_messages,
            stop_sequences=conf.display(messages)[1],
            pseudo_history_conf=conf,
        )
