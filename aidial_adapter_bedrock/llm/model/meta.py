from typing import Any, AsyncIterator, Dict, List, Optional

from aidial_sdk.chat_completion import Message
from typing_extensions import override

from aidial_adapter_bedrock.bedrock import (
    Bedrock,
    ResponseWithInvocationMetricsMixin,
)
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_model import PseudoChatModel
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_META
from aidial_adapter_bedrock.llm.model.llama.conf import LlamaConf
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)


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
    model: str
    client: Bedrock

    @classmethod
    def create(cls, client: Bedrock, model: str, conf: LlamaConf):
        return cls(
            client=client,
            model=model,
            tokenize_string=default_tokenize_string,
            tools_emulator=default_tools_emulator,
            chat_emulator=conf.chat_emulator,
            partitioner=conf.chat_partitioner,
        )

    @override
    def preprocess_messages(self, messages: List[Message]) -> List[Message]:
        messages = super().preprocess_messages(messages)

        # Llama behaves strangely on empty prompt:
        # it generate empty string, but claims to used up all available completion tokens.
        # So replace it with a single space.
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
            response, _headers = await self.client.ainvoke_non_streaming(
                self.model, args
            )
            stream = response_to_stream(response, usage)

        stream = self.post_process_stream(stream, params, self.chat_emulator)

        async for content in stream:
            consumer.append_content(content)
        consumer.close_content()

        consumer.add_usage(usage)
