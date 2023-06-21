from enum import Enum
from typing import Tuple

from llm.chat_emulation import ChatEmulationType
from utils.cli import select_option


class BedrockModels(str, Enum):
    amazon_titan_tg1_large = "amazon.titan-tg1-large"
    ai21_j2_grande_instruct = "ai21.j2-grande-instruct"
    ai21_j2_jumbo_instruct = "ai21.j2-jumbo-instruct"
    anthropic_claude_instant_v1 = "anthropic.claude-instant-v1"
    anthropic_claude_v1 = "anthropic.claude-v1"


def parse_args() -> Tuple[str, ChatEmulationType]:
    model_id = select_option(
        "Select the model", [e.name for e in BedrockModels]
    )
    emulation_type = select_option(
        "Select chat emulation type", [e.name for e in ChatEmulationType]
    )
    chat_emulation_type = ChatEmulationType[emulation_type]

    return model_id, chat_emulation_type
