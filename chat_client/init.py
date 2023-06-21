from typing import List, Tuple

from llm.chat_emulation import ChatEmulationType
from utils.cli import select_option


def parse_args() -> Tuple[str, ChatEmulationType]:
    models: List[str] = [
        "amazon.titan-tg1-large"
        # "amazon.titan-e1t-medium",  # Embeddings model, returns {"embedding": List[float], "inputTextTokenCount": int}
        # "stability.stable-diffusion-xl",  # Not relevant
        "ai21.j2-grande-instruct"
        "ai21.j2-jumbo-instruct"
        "anthropic.claude-instant-v1"
        "anthropic.claude-v1"
    ]

    model_id = select_option("Select the model", models)
    emulation_type = select_option(
        "Select chat emulation type", [e.name for e in ChatEmulationType]
    )
    chat_emulation_type = ChatEmulationType[emulation_type]

    return model_id, chat_emulation_type
