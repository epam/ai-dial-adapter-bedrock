from typing import List, Optional

from aidial_sdk.chat_completion import FinishReason

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.consumer import Attachment, Consumer


class CollectConsumer(Consumer):
    usage: TokenUsage
    content: str
    attachments: List[Attachment]
    discarded_messages: Optional[List[int]]

    def __init__(self):
        self.usage = TokenUsage()
        self.content = ""
        self.attachments = []
        self.discarded_messages = None

    def append_content(self, content: str):
        self.content += content

    def close_content(self, finish_reason: FinishReason | None = None):
        pass

    def add_attachment(self, attachment: Attachment):
        self.attachments.append(attachment)

    def add_usage(self, usage: TokenUsage):
        self.usage.accumulate(usage)

    def set_discarded_messages(self, discarded_messages: List[int]):
        self.discarded_messages = discarded_messages
