import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import boto3
from anthropic.tokenizer import count_tokens
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llm.chat_model import (
    ChatModel,
    Model,
    ModelResponse,
    ResponseData,
    TokenUsage,
)
from universal_api.request import ModelParameters
from utils.concurrency import make_async
from utils.log_config import bedrock_logger as log


class BedrockModelId(TypedDict):
    modelArn: str
    modelId: str


class IOutput(ABC):
    @abstractmethod
    def content(self) -> str:
        pass

    def data(self) -> list[ResponseData]:
        return []

    @abstractmethod
    def usage(self, prompt: str) -> TokenUsage:
        pass


class AmazonResult(BaseModel):
    tokenCount: int
    outputText: str
    completionReason: Optional[str]


class AmazonResponse(BaseModel, IOutput):
    inputTextTokenCount: int
    results: List[AmazonResult]

    def content(self) -> str:
        assert (
            len(self.results) == 1
        ), "AmazonResponse should only have one result"
        return self.results[0].outputText

    def usage(self, prompt: str) -> TokenUsage:
        assert (
            len(self.results) == 1
        ), "AmazonResponse should only have one result"
        return TokenUsage(
            prompt_tokens=self.inputTextTokenCount,
            completion_tokens=self.results[0].tokenCount,
        )


class AnthropicResponse(BaseModel, IOutput):
    completion: str
    stop_reason: str  # Literal["stop_sequence"]

    def content(self) -> str:
        return self.completion

    def usage(self, prompt: str) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=count_tokens(prompt),
            completion_tokens=count_tokens(self.completion),
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


class AI21Response(BaseModel, IOutput):
    id: int
    prompt: TextAndTokens
    completions: List[Completion]

    def content(self) -> str:
        assert (
            len(self.completions) == 1
        ), "AI21Response should only have one completion"
        return self.completions[0].data.text

    def usage(self, prompt: str) -> TokenUsage:
        assert (
            len(self.completions) == 1
        ), "AI21Response should only have one completion"
        return TokenUsage(
            prompt_tokens=len(self.prompt.tokens),
            completion_tokens=len(self.completions[0].data.tokens),
        )


class StabilityStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class StabilityError(BaseModel):
    id: str
    message: str
    name: str


class StabilityArtifact(BaseModel):
    seed: int
    base64: str
    finish_reason: str = Field(alias="finishReason")


class StabilityResponse(BaseModel, IOutput):
    # TODO: Use tagged union artifacts/error
    result: str
    artifacts: Optional[list[StabilityArtifact]]
    error: Optional[StabilityError]

    def content(self) -> str:
        self._throw_if_error()
        return ""

    def data(self) -> list[ResponseData]:
        self._throw_if_error()
        return [ResponseData(mime_type="image/png", name="image", content=self.artifacts[0].base64)]  # type: ignore

    def usage(self, prompt: str) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
        )

    def _throw_if_error(self):
        if self.result == StabilityStatus.ERROR:
            raise Exception(self.error.message)  # type: ignore


class TaggedAmazonResponse(AmazonResponse):
    provider: Literal["amazon"]


class TaggedAnthropicResponse(AnthropicResponse):
    provider: Literal["anthropic"]


class TaggedAI21Response(AI21Response):
    provider: Literal["ai21"]


class TaggedStabilityResponse(StabilityResponse):
    provider: Literal["stability"]


class BedrockResponse(BaseModel, IOutput):
    __root__: Annotated[
        Union[
            TaggedAmazonResponse,
            TaggedAnthropicResponse,
            TaggedAI21Response,
            TaggedStabilityResponse,
        ],
        Field(discriminator="provider"),
    ]

    def content(self) -> str:
        return self.__root__.content()

    def data(self) -> list[ResponseData]:
        return self.__root__.data()

    def usage(self, prompt: str) -> TokenUsage:
        return self.__root__.usage(prompt)


class BedrockModels:
    def __init__(self, region: str):
        session = boto3.Session()
        self.bedrock = session.client("bedrock", region)

    def models(self) -> List[BedrockModelId]:
        return self.bedrock.list_foundation_models()["modelSummaries"]


# Simplified copy of langchain.llms.bedrock.LLMInputOutputAdapter.prepare_input
def prepare_input(
    provider: str, prompt: str, model_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    input_body = {**model_kwargs}
    if provider == "anthropic" or provider == "ai21":
        input_body["prompt"] = prompt
    elif provider == "amazon":
        input_body = dict()
        input_body["inputText"] = prompt
        input_body["textGenerationConfig"] = {**model_kwargs}
    elif provider == "stability":
        input_body = dict()
        input_body["text_prompts"] = [{"text": prompt}]
    else:
        input_body["inputText"] = prompt

    return input_body


def prepare_model_kwargs(
    provider: str, model_params: ModelParameters
) -> Dict[str, Any]:
    model_kwargs = {}

    # NOTE: See https://docs.anthropic.com/claude/reference/complete_post
    if provider == "anthropic":
        if model_params.max_tokens is not None:
            model_kwargs["max_tokens_to_sample"] = model_params.max_tokens
        else:
            # The max tokens parameter is required for Anthropic models.
            # Choosing reasonable default.
            model_kwargs["max_tokens_to_sample"] = 500

        if model_params.stop is not None:
            model_kwargs["stop_sequences"] = model_params.stop

        if model_params.temperature is not None:
            model_kwargs["temperature"] = model_params.temperature

        # Doesn't have any effect. AWS always sends the whole response at once.
        # streaming = model_kwargs.streaming

        if model_params.top_p is not None:
            model_kwargs["top_p"] = model_params.top_p

        # OpenAI API doesn't have top_k parameter.
        # if model_params.top_k is not None:
        #    model_kwargs["top_k"] = model_params.top_k

    # NOTE: API See https://docs.ai21.com/reference/j2-instruct-ref
    # NOTE: Per-model token limits: https://docs.ai21.com/docs/choosing-the-right-instance-type-for-amazon-sagemaker-models#foundation-models
    if provider == "ai21":
        if model_params.max_tokens is not None:
            model_kwargs["maxTokens"] = model_params.max_tokens
        else:
            # The default for max tokens is 16, which is too small for most use cases
            model_kwargs["maxTokens"] = 500

        if model_params.temperature is not None:
            model_kwargs["temperature"] = model_params.temperature
        else:
            # The default AI21 temperature is 0.7.
            # The default OpenAI temperature is 1.0.
            # Choosing the OpenAI default since we pretend AI21 to be OpenAI.
            model_kwargs["temperature"] = 1.0

        if model_params.top_p is not None:
            model_kwargs["topP"] = model_params.top_p

        if model_params.stop is not None:
            model_kwargs["stopSequences"] = model_params.stop

        # NOTE: AI21 has "numResults" parameter, however we emulate multiple result
        # via mutliple calls to support all models uniformly.

    if provider == "amazon":
        if model_params.temperature is not None:
            model_kwargs["temperature"] = model_params.temperature
        # NOTE: There is no documentation for Amazon models currently.
        # NOTE: max tokens is 128 by default. The parameter name is not known.

    return model_kwargs


class BedrockAdapter(ChatModel):
    def __init__(
        self,
        model_id: str,
        model_provider: str,
        model_params: ModelParameters,
        model_kwargs: Dict[str, Any],
        bedrock: Any,
    ):
        self.model_id = model_id
        self.model_provider = model_provider
        self.model_params = model_params
        self.model_kwargs = model_kwargs
        self.bedrock = bedrock

    @classmethod
    async def create(
        cls, model_id: str, region: str, model_params: ModelParameters
    ) -> "BedrockAdapter":
        model_provider = Model.parse(model_id).provider

        model_kwargs = prepare_model_kwargs(model_provider, model_params)

        bedrock = await make_async(
            lambda _: boto3.Session().client("bedrock", region),
            (),
        )

        return cls(
            model_id, model_provider, model_params, model_kwargs, bedrock
        )

    async def acall(self, prompt: str) -> ModelResponse:
        return await make_async(self._call, prompt)

    def _call(self, prompt: str) -> ModelResponse:
        log.debug(f"prompt:\n{prompt}")

        model_response = self.bedrock.invoke_model(
            body=json.dumps(
                prepare_input(self.model_provider, prompt, self.model_kwargs)
            ),
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
        )

        body = json.loads(model_response["body"].read())
        resp = BedrockResponse.parse_obj(
            {"provider": self.model_provider, **body}
        )
        response = ModelResponse(
            content=resp.content(), data=resp.data(), usage=resp.usage(prompt)
        )

        log.debug(f"response:\n{response.json()}")
        return response
