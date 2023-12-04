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
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_META


class MetaResult(BaseModel):
    tokenCount: int
    outputText: str
    completionReason: Optional[str]


class MetaResponse(BaseModel):
    generation: str
    prompt_token_count: int
    generation_token_count: int
    stop_reason: str

    def content(self) -> str:
        return self.generation

    def usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_token_count,
            completion_tokens=self.generation_token_count,
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


async def response_to_stream(
    response: dict, usage: TokenUsage
) -> AsyncIterator[str]:
    resp = MetaResponse.parse_obj(response)
    usage.accumulate(resp.usage())
    yield resp.content()


class MetaAdapter(PseudoChatModel):
    client: Bedrock

    def __init__(
        self,
        client: Bedrock,
        model: str,
        count_tokens: Callable[[str], int],
        pseudo_history_conf: PseudoChatConf,
    ):
        super().__init__(model, count_tokens, pseudo_history_conf)
        self.client = client

    @override
    def _validate_and_cleanup_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        messages = super()._validate_and_cleanup_messages(messages)

        # Llama behave strangely on empty prompt:
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

        response = await self.client.ainvoke_non_streaming(self.model, args)
        stream = response_to_stream(response, usage)
        stream = self.post_process_stream(
            stream, params, self.pseudo_history_conf
        )

        async for content in stream:
            consumer.append_content(content)

        consumer.add_usage(usage)
