"""
Turning a chat into a prompt for the Llama3 model.

See as a reference:
https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
"""

from typing import List, Literal, Optional, Tuple, assert_never

from aidial_adapter_bedrock.llm.chat_emulator import ChatEmulator
from aidial_adapter_bedrock.llm.message import (
    AIRegularMessage,
    BaseMessage,
    HumanRegularMessage,
    SystemMessage,
)
from aidial_adapter_bedrock.llm.model.llama.conf import LlamaConf


def get_role(message: BaseMessage) -> Literal["system", "user", "assistant"]:
    match message:
        case SystemMessage():
            return "system"
        case HumanRegularMessage():
            return "user"
        case AIRegularMessage():
            return "assistant"
        case _:
            assert_never(message)


def encode_header(message: BaseMessage) -> str:
    ret = ""
    ret += "<|start_header_id|>"
    ret += get_role(message)
    ret += "<|end_header_id|>"
    ret += "\n\n"
    return ret


def encode_message(message: BaseMessage) -> str:
    ret = encode_header(message)
    ret += message.text_content
    ret += "<|eot_id|>"
    return ret


def encode_dialog_prompt(messages: List[BaseMessage]) -> str:
    ret = ""
    ret += "<|begin_of_text|>"
    for message in messages:
        ret += encode_message(message)
    ret += encode_header(AIRegularMessage(content=""))
    return ret


class LlamaChatEmulator(ChatEmulator):
    def display(self, messages: List[BaseMessage]) -> Tuple[str, List[str]]:
        return encode_dialog_prompt(messages), []

    def get_ai_cue(self) -> Optional[str]:
        return None


def llama3_chat_partitioner(messages: List[BaseMessage]) -> List[int]:
    return [1] * len(messages)


llama3_config = LlamaConf(
    chat_partitioner=llama3_chat_partitioner,
    chat_emulator=LlamaChatEmulator(),
)
