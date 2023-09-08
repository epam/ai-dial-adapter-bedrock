from enum import Enum
from typing import Tuple

from llm.chat_emulation.types import ChatEmulationType
from utils.cli import select_enum


class BedrockDeployment(str, Enum):
    AMAZON_TITAN_TG1_LARGE = "amazon.titan-tg1-large"
    AI21_J2_GRANDE_INSTRUCT = "ai21.j2-grande-instruct"
    AI21_J2_JUMBO_INSTRUCT = "ai21.j2-jumbo-instruct"
    AI21_J2_MID = "ai21.j2-mid"
    AI21_J2_ULTRA = "ai21.j2-ultra"
    ANTHROPIC_CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1"
    ANTHROPIC_CLAUDE_V1 = "anthropic.claude-v1"
    ANTHROPIC_CLAUDE_V2 = "anthropic.claude-v2"
    STABILITY_STABLE_DIFFUSION_XL = "stability.stable-diffusion-xl"

    def get_model_id(self) -> str:
        return self.value


def choose_deployment() -> Tuple[BedrockDeployment, ChatEmulationType]:
    deployment = select_enum("Select the deployment", BedrockDeployment)
    chat_emulation_type = select_enum(
        "Select chat emulation type", ChatEmulationType
    )

    return deployment, chat_emulation_type
