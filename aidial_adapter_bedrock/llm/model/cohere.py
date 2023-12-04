from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import override

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation.pseudo_chat import PseudoChatConf
from aidial_adapter_bedrock.llm.chat_model import PseudoChatModel
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_COHERE
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


class CohereResult(BaseModel):
    tokenCount: int
    outputText: str
    completionReason: Optional[str]


class Likelihood(BaseModel):
    likelihood: float
    token: str


class CohereGeneration(BaseModel):
    id: str
    text: str
    likelihood: float
    finish_reason: str
    token_likelihoods: List[Likelihood]


class CohereResponse(BaseModel):
    id: str
    prompt: Optional[str]
    generations: List[CohereGeneration]

    def content(self) -> str:
        return self.generations[0].text

    @property
    def tokens(self) -> List[str]:
        """Includes prompt and completion tokens"""
        return [lh.token for lh in self.generations[0].token_likelihoods]

    def usage(self) -> TokenUsage:
        special_tokens = 2
        total_tokens = len(self.tokens) - special_tokens

        # The structure for the tokens:
        # ["<BOS_TOKEN>", "User", ":", *<prompt>, "\n", "Chat", "bot", ":", "<EOP_TOKEN>", *<completion>]
        separator = "<EOP_TOKEN>"
        if separator in self.tokens:
            prompt_tokens = self.tokens.index(separator) + 1 - special_tokens
        else:
            log.error(f"Separator '{separator}' not found in tokens")
            prompt_tokens = total_tokens // 2

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=total_tokens - prompt_tokens,
        )


def convert_params(params: ModelParameters) -> Dict[str, Any]:
    ret = {}

    if params.temperature is not None:
        ret["temperature"] = params.temperature

    if params.max_tokens is not None:
        ret["max_tokens"] = params.max_tokens
    else:
        # Choosing reasonable default
        ret["max_tokens"] = DEFAULT_MAX_TOKENS_COHERE

    ret["return_likelihoods"] = "ALL"

    # NOTE: num_generations is supported

    return ret


def create_request(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **params}


async def response_to_stream(
    response: dict, usage: TokenUsage
) -> AsyncIterator[str]:
    resp = CohereResponse.parse_obj(response)
    usage.accumulate(resp.usage())

    log.debug(f"tokens :\n{''.join(resp.tokens)}")

    yield resp.content()


class CohereAdapter(PseudoChatModel):
    client: Bedrock

    def __init__(
        self,
        client: Bedrock,
        model_id: str,
        count_tokens: Callable[[str], int],
        pseudo_history_conf: PseudoChatConf,
    ):
        super().__init__(model_id, count_tokens, pseudo_history_conf)
        self.client = client

    @override
    def _validate_and_cleanup_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        messages = super()._validate_and_cleanup_messages(messages)

        # Cohere doesn't support empty messages,
        # so replace it with a single space.
        for msg in messages:
            msg.content = msg.content or " "

        return messages

    async def _apredict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt, convert_params(params))
        response = await self.client.ainvoke_non_streaming(self.model, args)

        usage = TokenUsage()
        stream = response_to_stream(response, usage)
        stream = self.post_process_stream(
            stream, params, self.pseudo_history_conf
        )

        async for content in stream:
            consumer.append_content(content)

        consumer.add_usage(usage)
