from enum import Enum
from typing import Callable, List, Optional, Set, Tuple

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
    SystemMessage,
)
from aidial_adapter_bedrock.utils.list import exclude_indices


class RolePrompt(str, Enum):
    HUMAN = "\n\nHuman:"
    ASSISTANT = "\n\nAssistant:"


STOP_SEQUENCES: List[str] = [RolePrompt.HUMAN]


PRELUDE = f"""
You are a helpful assistant participating in a dialog with a user.
The messages from the user start with "{RolePrompt.HUMAN.strip()}".
The messages from you start with "{RolePrompt.ASSISTANT.strip()}".
Reply to the last message from the user taking into account the preceding dialog history.
====================
""".strip()


def _format_message(message: BaseMessage) -> str:
    role = (
        RolePrompt.HUMAN
        if isinstance(message, (SystemMessage, HumanMessage))
        else RolePrompt.ASSISTANT
    )
    return (role + " " + message.content.lstrip()).rstrip()


class PseudoChatHistory(History):
    stop_sequences: List[str]

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
                        ]
                    ),
                    len(discarded_messages),
                )

        if discarded_messages:
            discarded_messages_count = len(discarded_messages)
            if (
                source_messages_count - discarded_messages_count == 1
                and isinstance(last_source_message, HumanMessage)
            ):
                history = PseudoChatHistory.create([last_source_message])
                prompt_tokens = sum(
                    count_tokens(message.text) for message in history.messages
                )
                if prompt_tokens <= max_prompt_tokens:
                    return history, len(discarded_messages)

            raise ValidationError(
                f"The token size of system messages and the last user message ({prompt_tokens}) exceeds"
                f" prompt token limit ({max_prompt_tokens})."
            )

        raise ValidationError(
            f"Prompt token size ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})."
        )

    @classmethod
    def create(cls, messages: List[BaseMessage]) -> "PseudoChatHistory":
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            single_message = messages[0]
            return cls(
                messages=[
                    FormattedMessage(
                        text=single_message.content,
                        source_message=single_message,
                    )
                ],
                stop_sequences=[],
            )

        formatted_messages = [FormattedMessage(text=PRELUDE)]

        for index, message in enumerate(messages):
            formatted_messages.append(
                FormattedMessage(
                    text=_format_message(message),
                    source_message=message,
                    is_important=is_important_message(messages, index),
                )
            )

        formatted_messages.append(
            FormattedMessage(text=_format_message(AIMessage(content="")))
        )

        return cls(messages=formatted_messages, stop_sequences=STOP_SEQUENCES)
