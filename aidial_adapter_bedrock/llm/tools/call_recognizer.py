from typing import Callable

from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
)

CallParser = Callable[[str], AIToolCallMessage | AIFunctionCallMessage | None]


class CallRecognizer:
    start_tag: str
    call_parser: CallParser

    acc: str

    def __init__(self, start_tag: str, call_parser: CallParser):
        self.start_tag = start_tag
        self.call_parser = call_parser

        self.acc = ""

    def consume_chunk(
        self, chunk: str | None
    ) -> str | AIToolCallMessage | AIFunctionCallMessage | None:
        if chunk is None:
            """End of the chunk stream"""
            if self.start_tag in self.acc:
                return self.call_parser(self.acc)
            else:
                return self.acc

        self.acc += chunk
