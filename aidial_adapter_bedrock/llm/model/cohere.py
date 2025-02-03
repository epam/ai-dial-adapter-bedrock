from typing import Any, AsyncIterator, Dict, List, Optional

from aidial_sdk.chat_completion import Message
from pydantic import BaseModel, Field

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
    ChatCompletionAdapter,
    TextCompletionAdapter,
    default_preprocess_messages,
    keep_last_and_system_messages,
    trivial_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.decorator.base import compose_decorators
from aidial_adapter_bedrock.llm.decorator.preprocess_messages import (
    preprocess_messages_decorator,
)
from aidial_adapter_bedrock.llm.decorator.pseudo_chat import pseudo_chat_adapter
from aidial_adapter_bedrock.llm.decorator.replicator import replicator_decorator
from aidial_adapter_bedrock.llm.decorator.tools_emulator import (
    tools_emulator_decorator,
)
from aidial_adapter_bedrock.llm.decorator.truncate_prompt import (
    truncate_prompt_decorator,
)
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_COHERE
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)
from aidial_adapter_bedrock.utils.list_projection import ListProjection


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


def _preprocess_cohere_messages(
    messages: List[Message],
) -> ListProjection[Message]:
    ret = default_preprocess_messages(messages)

    # Cohere doesn't support empty messages,
    # so replace it with a single space.
    for msg in ret.raw_list:
        msg.content = msg.content or " "

    return ret


def create_adapter(client: Bedrock, model: str) -> ChatCompletionAdapter:
    return compose_decorators(
        preprocess_messages_decorator(_preprocess_cohere_messages),
        truncate_prompt_decorator(
            keep_message=keep_last_and_system_messages,
            partitioner=trivial_partitioner,
        ),
        replicator_decorator(),
        tools_emulator_decorator(default_tools_emulator),
    )(
        pseudo_chat_adapter(cohere_emulator)(
            CohereAdapter(client=client, model=model)
        )
    )


class CohereAdapter(TextCompletionAdapter):
    model: str
    client: Bedrock

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

        stream = post_process_completion_stream(params, cohere_emulator, stream)

        async for content in stream:
            consumer.append_content(content)
        consumer.close_content()

        consumer.add_usage(usage)

    async def count_completion_tokens(self, string: str) -> int:
        return default_tokenize_string(string)

    async def count_prompt_tokens(
        self, params: ModelParameters, prompt: str
    ) -> int:
        return default_tokenize_string(prompt)
