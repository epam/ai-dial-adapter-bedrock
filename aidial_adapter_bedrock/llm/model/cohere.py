from typing import Any, AsyncIterator, Dict, List, Optional

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel, Field
from typing_extensions import override

from aidial_adapter_bedrock.bedrock import (
    Bedrock,
    Headers,
    ResponseWithInvocationMetricsMixin,
    usage_from_headers,
)
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulator import (
    BasicChatEmulator,
    CueMapping,
    post_process_completion_stream,
)
from aidial_adapter_bedrock.llm.chat_model import (
    PseudoChatModel,
    trivial_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_COHERE
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)


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


class CohereResponse(ResponseWithInvocationMetricsMixin):
    id: str
    prompt: Optional[str]
    generations: List[CohereGeneration]

    def content(self) -> str:
        return self.generations[0].text

    @property
    def tokens(self) -> List[str]:
        """Includes prompt and completion tokens"""
        return [lh.token for lh in self.generations[0].token_likelihoods]


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
        usage.accumulate(resp.usage_from_metrics())
        yield resp.content()


async def response_to_stream(
    response_body: dict, response_headers: Headers, usage: TokenUsage
) -> AsyncIterator[str]:
    resp = CohereResponse.parse_obj(response_body)
    usage.accumulate(usage_from_headers(response_headers))
    yield resp.content()


cohere_emulator = BasicChatEmulator(
    prelude_template=None,
    should_prefix_with_cue=lambda _, idx: idx > 0,
    should_add_invitation_cue=False,
    should_fallback_to_completion=False,
    cues=CueMapping(
        system="User:",
        human="User:",
        ai="Chatbot:",
    ),
    separator="\n",
)


class CohereAdapter(PseudoChatModel):
    model: str
    client: Bedrock

    @classmethod
    def create(cls, client: Bedrock, model: str):
        return cls(
            client=client,
            model=model,
            tokenize_string=default_tokenize_string,
            chat_emulator=cohere_emulator,
            tools_emulator=default_tools_emulator,
            partitioner=trivial_partitioner,
        )

    @override
    def preprocess_messages(self, messages: List[Message]) -> List[Message]:
        messages = super().preprocess_messages(messages)

        # Cohere doesn't support empty messages,
        # so replace it with a single space.
        for msg in messages:
            msg.content = msg.content or " "

        return messages

    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt, convert_params(params))

        usage = TokenUsage()

        if params.stream:
            chunks = self.client.ainvoke_streaming(self.model, args)
            stream = chunks_to_stream(chunks, usage)
        else:
            response, headers = await self.client.ainvoke_non_streaming(
                self.model, args
            )
            stream = response_to_stream(response, headers, usage)

        stream = post_process_completion_stream(
            params, self.chat_emulator, stream
        )

        async for content in stream:
            consumer.append_content(content)
        consumer.close_content()

        consumer.add_usage(usage)
