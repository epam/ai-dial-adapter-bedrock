from abc import ABC, abstractmethod
from typing import List, Optional

from aidial_sdk.chat_completion import Choice, FunctionCall
from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.tools.claude import parse_function_call


class Attachment(BaseModel):
    type: str | None = None
    title: str | None = None
    data: str | None = None
    url: str | None = None
    reference_url: str | None = None
    reference_type: str | None = None


class Consumer(ABC):
    @abstractmethod
    def append_content(self, content: str):
        pass

    @abstractmethod
    def add_attachment(self, attachment: Attachment):
        pass

    @abstractmethod
    def add_usage(self, usage: TokenUsage):
        pass

    @abstractmethod
    def set_discarded_messages(self, discarded_messages: int):
        pass


class ChoiceConsumer(Consumer):
    usage: TokenUsage
    choice: Choice
    discarded_messages: Optional[int]

    def __init__(self, choice: Choice):
        self.choice = choice
        self.usage = TokenUsage()
        self.discarded_messages = None

    def append_content(self, content: str):
        # FIXME: works only in non-streaming mode
        call: Optional[FunctionCall] = None
        try:
            call = parse_function_call(content)
        except Exception:
            pass

        if call is not None:
            # FIXME: check the functions/tools mode
            self.choice.add_function_call(call)
        else:
            self.choice.append_content(content)

    def add_attachment(self, attachment: Attachment):
        self.choice.add_attachment(**attachment.dict())

    def add_usage(self, usage: TokenUsage):
        self.usage.accumulate(usage)

    def set_discarded_messages(self, discarded_messages: int):
        self.discarded_messages = discarded_messages


class CollectConsumer(Consumer):
    usage: TokenUsage
    content: str
    attachments: List[Attachment]
    discarded_messages: Optional[int]

    def __init__(self):
        self.usage = TokenUsage()
        self.content = ""
        self.attachments = []
        self.discarded_messages = None

    def append_content(self, content: str):
        self.content += content

    def add_attachment(self, attachment: Attachment):
        self.attachments.append(attachment)

    def add_usage(self, usage: TokenUsage):
        self.usage.accumulate(usage)

    def set_discarded_messages(self, discarded_messages: int):
        self.discarded_messages = discarded_messages
