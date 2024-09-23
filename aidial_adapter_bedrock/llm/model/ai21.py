from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulator import default_emulator
from aidial_adapter_bedrock.llm.chat_model import (
    PseudoChatModel,
    trivial_partitioner,
)
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_AI21
from aidial_adapter_bedrock.llm.tokenize import default_tokenize_string
from aidial_adapter_bedrock.llm.tools.default_emulator import (
    default_tools_emulator,
)


class TextRange(BaseModel):
    start: int
    end: int


class GeneratedToken(BaseModel):
    token: str
    logprob: float
    raw_logprob: float


class Token(BaseModel):
    generatedToken: GeneratedToken
    topTokens: Optional[Any]
    textRange: TextRange


class TextAndTokens(BaseModel):
    text: str
    tokens: List[Token]


class FinishReason(BaseModel):
    reason: str  # Literal["length", "endoftext"]
    length: Optional[int]


class Completion(BaseModel):
    data: TextAndTokens
    finishReason: FinishReason


class AI21Response(BaseModel):
    id: int
    prompt: TextAndTokens
    completions: List[Completion]

    def content(self) -> str:
        assert (
            len(self.completions) == 1
        ), "AI21Response should only have one completion"
        return self.completions[0].data.text

    def usage(self) -> TokenUsage:
        assert (
            len(self.completions) == 1
        ), "AI21Response should only have one completion"
        return TokenUsage(
            prompt_tokens=len(self.prompt.tokens),
            completion_tokens=len(self.completions[0].data.tokens),
        )


# NOTE: See https://docs.ai21.com/reference/j2-instruct-ref
def convert_params(params: ModelParameters) -> Dict[str, Any]:
    ret = {}

    if params.max_tokens is not None:
        ret["maxTokens"] = params.max_tokens
    else:
        # The default for max tokens is 16, which is too small for most use cases.
        # Choosing reasonable default.
        ret["maxTokens"] = DEFAULT_MAX_TOKENS_AI21

    if params.temperature is not None:
        #   AI21 temperature ranges from 0.0 to 1.0
        # OpenAI temperature ranges from 0.0 to 2.0
        # Thus scaling down by 2x to match the AI21 range
        ret["temperature"] = params.temperature / 2.0

    if params.top_p is not None:
        ret["topP"] = params.top_p

    if params.stop:
        ret["stopSequences"] = params.stop

    # NOTE: AI21 has "numResults" parameter, however we emulate multiple result
    # via multiple calls to support all models uniformly.

    return ret


def create_request(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **params}


class AI21Adapter(PseudoChatModel):
    model: str
    client: Bedrock

    @classmethod
    def create(cls, client: Bedrock, model: str):
        return cls(
            client=client,
            model=model,
            tokenize_string=default_tokenize_string,
            tools_emulator=default_tools_emulator,
            chat_emulator=default_emulator,
            partitioner=trivial_partitioner,
        )

    async def predict(
        self, consumer: Consumer, params: ModelParameters, prompt: str
    ):
        args = create_request(prompt, convert_params(params))
        response, _headers = await self.client.ainvoke_non_streaming(
            self.model, args
        )

        resp = AI21Response.parse_obj(response)

        consumer.append_content(resp.content())
        consumer.close_content()
        consumer.add_usage(resp.usage())
