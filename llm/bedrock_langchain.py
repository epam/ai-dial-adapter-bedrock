import logging
from typing import Optional, Tuple

from langchain.llms.bedrock import Bedrock

from llm.chat_model import ChatModel, TokenUsage
from utils.token_counter import get_num_tokens

log = logging.getLogger("bedrock")


def compute_usage(prompt: str, completion: str) -> TokenUsage:
    prompt_tokens = get_num_tokens(prompt)
    completion_tokens = get_num_tokens(completion)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class BedrockLangChain(ChatModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: Optional[int],
        region: str = "us-east-1",
    ):
        self.model_id = model_id
        provider = model_id.split(".")[0]

        model_kwargs = {}
        if provider == "anthropic":
            model_kwargs["max_tokens_to_sample"] = (
                max_tokens if max_tokens is not None else 500
            )

        self.model = Bedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs=model_kwargs,
        )  # type: ignore

    def _call(self, prompt: str) -> Tuple[str, TokenUsage]:
        log.debug(f"prompt:\n{prompt}")
        response = self.model._call(prompt)
        log.debug(f"response:\n{response}")
        return response, compute_usage(prompt, response)
