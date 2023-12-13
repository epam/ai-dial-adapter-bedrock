from typing import Any, AsyncIterator, Dict, List, Optional

from typing_extensions import override

from aidial_adapter_bedrock.bedrock import (
    Bedrock,
    ResponseWithInvocationMetricsMixin,
)
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import PseudoChatModel
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.message import BaseMessage
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_META
from aidial_adapter_bedrock.llm.model.llama_chat import (
    llama_emulator,
    llama_partitioner,
)
from aidial_adapter_bedrock.llm.tokenize import default_tokenize
from aidial_adapter_bedrock.llm.tools.emulator import default_tools_emulator


class MetaResponse(ResponseWithInvocationMetricsMixin):
    generation: str
    prompt_token_count: Optional[int]
    generation_token_count: Optional[int]
    stop_reason: Optional[str]

    def content(self) -> str:
        return self.generation

    def usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_token_count or 0,
            completion_tokens=self.generation_token_count or 0,
        )


def convert_params(params: ModelParameters) -> Dict[str, Any]:
    ret = {}

    if params.temperature is not None:
        ret["temperature"] = params.temperature

    if params.top_p is not None:
        ret["top_p"] = params.top_p

    if params.max_tokens is not None:
        ret["max_gen_len"] = params.max_tokens
    else:
        # Choosing reasonable default
        ret["max_gen_len"] = DEFAULT_MAX_TOKENS_META

    return ret


def create_request(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **params}


async def chunks_to_stream(
    chunks: AsyncIterator[dict], usage: TokenUsage
) -> AsyncIterator[str]:
    async for chunk in chunks:
        resp = MetaResponse.parse_obj(chunk)
        usage.accumulate(resp.usage_by_metrics())
        yield resp.content()


async def response_to_stream(
    response: dict, usage: TokenUsage
) -> AsyncIterator[str]:
    resp = MetaResponse.parse_obj(response)
    usage.accumulate(resp.usage())
    yield resp.content()


class MetaAdapter(PseudoChatModel):
    client: Bedrock

    @classmethod
    def create(cls, client: Bedrock, model: str):
        return cls(
            client=client,
            model=model,
            tokenize=default_tokenize,
            chat_emulator=llama_emulator,
            tools_emulator=default_tools_emulator,
            partitioner=llama_partitioner,
        )

    @override
    def _validate_and_cleanup_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        messages = super()._validate_and_cleanup_messages(messages)

        # Llama behaves strangely on empty prompt:
        # it generate empty string, but claims to used up all available completion tokens.
        # So replace it with a single space.
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
        consumer.close_content()

        consumer.add_usage(usage)
