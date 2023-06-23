import logging
import re
from typing import List, Optional, Tuple, TypedDict

from langchain.llms.bedrock import Bedrock
from langchain.schema import BaseMessage

from llm.chat_emulation import emulate_chat, meta_chat
from llm.chat_emulation.types import ChatEmulationType
from utils.token_counter import get_num_tokens

log = logging.getLogger("bedrock")


def create_model(model_id: str, max_tokens: Optional[int]) -> Bedrock:
    provider = model_id.split(".")[0]

    model_kwargs = {}
    if provider == "anthropic":
        model_kwargs["max_tokens_to_sample"] = (
            max_tokens if max_tokens is not None else 500
        )

    return Bedrock(
        model_id=model_id,
        region_name="us-east-1",
        model_kwargs=model_kwargs,
    )  # type: ignore


# Copy of langchain.llms.utils::enforce_stop_tokens with a bugfix: stop words are escaped.
def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    stop_escaped = [re.escape(s) for s in stop]
    return re.split("|".join(stop_escaped), text)[0]


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def compute_usage(prompt: str, completion: str) -> TokenUsage:
    prompt_tokens = get_num_tokens(prompt)
    completion_tokens = get_num_tokens(completion)
    total_tokens = prompt_tokens + completion_tokens
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def chat(
    model: Bedrock,
    chat_emulation_type: ChatEmulationType,
    history: List[BaseMessage],
) -> Tuple[str, TokenUsage]:
    stop: Optional[List[str]] = None
    if chat_emulation_type == ChatEmulationType.META_CHAT:
        stop = [meta_chat.stop]

    prompt = emulate_chat(model.model_id, chat_emulation_type, history)
    log.debug(f"prompt:\n{prompt}")
    response = model._call(prompt)

    # Langchain has a bug in enforce_stop_tokens, so we have to reimplement it here.
    if stop is not None:
        response = enforce_stop_tokens(response, stop)

    log.debug(f"response:\n{response}")
    return response, compute_usage(prompt, response)


def completion(model: Bedrock, prompt: str) -> Tuple[str, TokenUsage]:
    log.debug(f"prompt:\n{prompt}")
    response = model._call(prompt, stop=None)
    log.debug(f"response:\n{response}")
    return response, compute_usage(prompt, response)
