from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel
from typing_extensions import override

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation.chat_emulator import ChatEmulator
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


# Simplified version of https://github.com/huggingface/transformers/blob/c99f25476312521d4425335f970b198da42f832d/src/transformers/models/llama/tokenization_llama.py#L415
# See also for the reference: https://github.com/facebookresearch/llama/blob/556949fdfb72da27c2f4a40b7f0e4cf0b8153a28/llama/generation.py#L320C9-L320C9
# See also: https://github.com/huggingface/transformers/blob/c99f25476312521d4425335f970b198da42f832d/src/transformers/models/llama/tokenization_llama.py#L415
def get_llama2_chat_prompt(
    system_message: Optional[str],
    turns: List[Tuple[str, str]],
    message: str,
) -> str:
    ret: List[str] = []

    if system_message is not None and system_message.strip():
        ret.append(f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n")

    is_first_turn = True
    for human, assistant in turns:
        human = human if is_first_turn else human.strip()
        is_first_turn = False

        ret.append(f"{human} [/INST] {assistant.strip()} </s><s>[INST] ")

    message = message.strip() if is_first_turn else message
    ret.append(f"{message} [/INST]")

    return "".join(ret)


class MetaAdapter(PseudoChatModel):
    client: Bedrock

    def __init__(
        self,
        client: Bedrock,
        model: str,
        tokenize: Callable[[str], int],
        chat_emulator: ChatEmulator,
    ):
        super().__init__(model, tokenize, chat_emulator)
        self.client = client

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

        response = await self.client.ainvoke_non_streaming(self.model, args)
        stream = response_to_stream(response, usage)
        stream = self.post_process_stream(stream, params, self.chat_emulator)

        async for content in stream:
            consumer.append_content(content)

        consumer.add_usage(usage)
