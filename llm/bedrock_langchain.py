import logging
from typing import Tuple

from langchain.llms.bedrock import Bedrock

from llm.bedrock_custom import prepare_model_kwargs
from llm.chat_model import ChatModel, TokenUsage
from open_ai.types import CompletionParameters
from utils.token_counter import get_num_tokens

log = logging.getLogger("bedrock")


def compute_usage_estimation(prompt: str, completion: str) -> TokenUsage:
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
        model_params: CompletionParameters,
        region: str,
    ):
        self.model_id = model_id
        self.model_params = model_params

        provider = model_id.split(".")[0]

        model_kwargs = prepare_model_kwargs(provider, model_params)

        self.model = Bedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs=model_kwargs,
        )  # type: ignore

    def _call(self, prompt: str) -> Tuple[str, TokenUsage]:
        log.debug(f"prompt:\n{prompt}")
        response = self.model._call(prompt)
        log.debug(f"response:\n{response}")
        return response, compute_usage_estimation(prompt, response)
