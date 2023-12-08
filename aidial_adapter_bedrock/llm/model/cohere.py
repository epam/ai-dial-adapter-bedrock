from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import override

from aidial_adapter_bedrock.bedrock import Bedrock, InvocationMetrics
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation.chat_emulator import (
    BasicChatEmulator,
    CueMapping,
)
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
    token_likelihoods: List[Likelihood] = Field(repr=False)


class CohereResponse(BaseModel):
    id: str
    prompt: Optional[str]
    generations: List[CohereGeneration]
    invocation_metrics: Optional[InvocationMetrics] = Field(
        alias="amazon-bedrock-invocationMetrics"
    )

    def content(self) -> str:
        return self.generations[0].text

    @property
    def tokens(self) -> List[str]:
        """Includes prompt and completion tokens"""
        return [lh.token for lh in self.generations[0].token_likelihoods]

    def usage_by_metrics(self) -> TokenUsage:
        metrics = self.invocation_metrics
        if metrics is None:
            return TokenUsage()

        return metrics.to_usage()

    def usage_by_tokens(self) -> TokenUsage:
        special_tokens = 7
        total_tokens = len(self.tokens) - special_tokens

        # The structure for the response:
        # ["<BOS_TOKEN>", "User", ":", *<prompt>, "\n", "Chat", "bot", ":", "<EOP_TOKEN>", *<completion>]
        # prompt_tokens = len(<prompt>)
        # completion_tokens = len(["<EOP_TOKEN>"] + <completion>)

        separator = "<EOP_TOKEN>"
        if separator in self.tokens:
            prompt_tokens = self.tokens.index(separator) - special_tokens
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


async def chunks_to_stream(
    chunks: AsyncIterator[dict], usage: TokenUsage
) -> AsyncIterator[str]:
    async for chunk in chunks:
        resp = CohereResponse.parse_obj(chunk)
        usage.accumulate(resp.usage_by_metrics())
        log.debug(f"tokens: {'|'.join(resp.tokens)!r}")
        yield resp.content()


async def response_to_stream(
    response: dict, usage: TokenUsage
) -> AsyncIterator[str]:
    resp = CohereResponse.parse_obj(response)
    usage.accumulate(resp.usage_by_tokens())
    log.debug(f"tokens: {'|'.join(resp.tokens)!r}")
    yield resp.content()


cohere_emulator = BasicChatEmulator(
    prelude_template=None,
    add_cue=lambda _, idx: idx > 0,
    add_invitation_cue=False,
    fallback_to_completion=False,
    cues=CueMapping(
        system="User:",
        human="User:",
        ai="Chatbot:",
    ),
    separator="\n",
)


class CohereAdapter(PseudoChatModel):
    client: Bedrock

    def __init__(
        self,
        client: Bedrock,
        model: str,
        tokenize: Callable[[str], int],
    ):
        super().__init__(model, tokenize, cohere_emulator)
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

        usage = TokenUsage()

        if params.stream:
            chunks = self.client.ainvoke_streaming(self.model, args)
            stream = chunks_to_stream(chunks, usage)
        else:
            response = await self.client.ainvoke_non_streaming(self.model, args)
            stream = response_to_stream(response, usage)

        stream = self.post_process_stream(stream, params, self.chat_emulator)

        async for content in stream:
            consumer.append_content(content)

        consumer.add_usage(usage)