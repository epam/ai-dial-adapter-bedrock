from typing import Callable

from aidial_adapter_bedrock.llm.message import (
    AIFunctionCallMessage,
    AIToolCallMessage,
)

CallParser = Callable[[str], AIToolCallMessage | AIFunctionCallMessage | None]


class CallRecognizer:
    init_buffer: int
    start_tag: str
    call_parser: CallParser

    acc: str
    found_tag: bool
    init_stage: bool

    def __init__(
        self, init_buffer: int, start_tag: str, call_parser: CallParser
    ):
        self.init_buffer = init_buffer
        self.start_tag = start_tag
        self.call_parser = call_parser

        self.acc = ""
        self.found_tag = False
        self.init_stage = True

    def consume_chunk(
        self, chunk: str | None
    ) -> str | AIToolCallMessage | AIFunctionCallMessage | None:
        if chunk is None:
            """End of the chunk stream"""
            if self.found_tag:
                return self.call_parser(self.acc)
            else:
                return self.acc

        self.acc += chunk

        if self.init_stage and len(self.acc) >= self.init_buffer:
            self.init_stage = False
            self.found_tag = self.start_tag in self.acc

        if not self.init_stage and not self.found_tag:
            ret = self.acc
            self.acc = ""
            return ret
