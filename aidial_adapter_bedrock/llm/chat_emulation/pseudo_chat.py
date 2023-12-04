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
    system: str
    human: str
    ai: str


class PseudoChatConf(BaseModel):
    prelude_template: str
    mapping: RoleMapping
    separator: str

    @property
    def prelude(self) -> str:
        return self.prelude_template.format(**self.mapping)

    @property
    def stop_sequences(self) -> List[str]:
        return [self.separator + self.mapping["human"]]

    def format_message(self, message: BaseMessage) -> str:
        role = self.mapping.get(message.type)
        if role is None:
            raise ValueError(f"Unknown message type: {message.type}")
        return (self.separator + role + " " + message.content.lstrip()).rstrip()


std_conf = PseudoChatConf(
    prelude_template="""
You are a helpful assistant participating in a dialog with a user.
The messages from the user start with "{ai}".
The messages from you start with "{human}".
Reply to the last message from the user taking into account the preceding dialog history.
====================
""".strip(),
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
                        pseudo_history_conf=self.pseudo_history_conf,
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
                    pseudo_history_conf=self.pseudo_history_conf,
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
        cls,
        messages: List[BaseMessage],
        pseudo_history_conf: PseudoChatConf = std_conf,
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
                pseudo_history_conf=pseudo_history_conf,
            )

        formatted_messages = [
            FormattedMessage(text=pseudo_history_conf.prelude)
        ]

        for index, message in enumerate(messages):
            formatted_messages.append(
                FormattedMessage(
                    text=pseudo_history_conf.format_message(message),
                    source_message=message,
                    is_important=is_important_message(messages, index),
                )
            )

        formatted_messages.append(
            FormattedMessage(
                text=pseudo_history_conf.format_message(AIMessage(content=""))
            )
        )

        return cls(
            messages=formatted_messages,
            stop_sequences=pseudo_history_conf.stop_sequences,
            pseudo_history_conf=pseudo_history_conf,
        )
