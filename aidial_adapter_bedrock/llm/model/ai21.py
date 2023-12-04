import json
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from aidial_adapter_bedrock.dial_api.request import ModelParameters
from aidial_adapter_bedrock.dial_api.token_usage import TokenUsage
from aidial_adapter_bedrock.llm.chat_emulation.pseudo_chat import PseudoChatConf
from aidial_adapter_bedrock.llm.chat_model import PseudoChatModel
from aidial_adapter_bedrock.llm.consumer import Consumer
from aidial_adapter_bedrock.llm.model.conf import DEFAULT_MAX_TOKENS_AI21
from aidial_adapter_bedrock.utils.concurrency import make_async


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
def prepare_model_kwargs(model_params: ModelParameters) -> Dict[str, Any]:
    model_kwargs = {}

    if model_params.max_tokens is not None:
        model_kwargs["maxTokens"] = model_params.max_tokens
    else:
        # The default for max tokens is 16, which is too small for most use cases.
        # Choosing reasonable default.
        model_kwargs["maxTokens"] = DEFAULT_MAX_TOKENS_AI21

    if model_params.temperature is not None:
        #   AI21 temperature ranges from 0.0 to 1.0
        # OpenAI temperature ranges from 0.0 to 2.0
        # Thus scaling down by 2x to match the AI21 range
        model_kwargs["temperature"] = model_params.temperature / 2.0

    if model_params.top_p is not None:
        model_kwargs["topP"] = model_params.top_p

    if model_params.stop:
        model_kwargs["stopSequences"] = model_params.stop

    # NOTE: AI21 has "numResults" parameter, however we emulate multiple result
    # via multiple calls to support all models uniformly.

    return model_kwargs


def prepare_input(prompt: str, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {"prompt": prompt, **model_kwargs}


class AI21Adapter(PseudoChatModel):
    def __init__(
        self,
        bedrock: Any,
        model_id: str,
        count_tokens: Callable[[str], int],
        pseudo_history_conf: PseudoChatConf,
    ):
        super().__init__(model_id, count_tokens, pseudo_history_conf)
        self.bedrock = bedrock

    async def _apredict(
        self, consumer: Consumer, model_params: ModelParameters, prompt: str
    ):
        await make_async(
            lambda args: self._call(*args), (consumer, model_params, prompt)
        )

    def _call(
        self, consumer: Consumer, model_params: ModelParameters, prompt: str
    ):
        model_kwargs = prepare_model_kwargs(model_params)

        model_response = self.bedrock.invoke_model(
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(prepare_input(prompt, model_kwargs)),
        )

        body = json.loads(model_response["body"].read())
        resp = AI21Response.parse_obj(body)

        consumer.append_content(resp.content())
        consumer.add_usage(resp.usage())
