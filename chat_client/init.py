from enum import Enum
from typing import Tuple

from llm.chat_emulation import ChatEmulationType
from utils.cli import select_enum


class BedrockModels(str, Enum):
    AMAZON_TITAN_TG1_LARGE = "amazon.titan-tg1-large"
    AI21_J2_GRANDE_INSTRUCT = "ai21.j2-grande-instruct"
    AI21_J2_JUMBO_INSTRUCT = "ai21.j2-jumbo-instruct"
    ANTHROPIC_CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1"
    ANTHROPIC_CLAUDE_V1 = "anthropic.claude-v1"


def parse_args() -> Tuple[BedrockModels, ChatEmulationType]:
    model_id = select_enum("Select the model", BedrockModels)
    chat_emulation_type = select_enum(
        "Select chat emulation type", ChatEmulationType
    )

    return model_id, chat_emulation_type
