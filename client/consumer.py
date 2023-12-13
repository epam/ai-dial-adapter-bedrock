from typing import List, Optional

from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.consumer import Attachment, Consumer


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
        # TODO: support recognition of tools/function calls
        self.content += content

    def add_attachment(self, attachment: Attachment):
        self.attachments.append(attachment)

    def add_usage(self, usage: TokenUsage):
        self.usage.accumulate(usage)

    def set_discarded_messages(self, discarded_messages: int):
        self.discarded_messages = discarded_messages
