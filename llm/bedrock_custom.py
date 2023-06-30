import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import boto3
from anthropic.tokenizer import count_tokens
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llm.chat_model import ChatModel, TokenUsage
from open_ai.types import CompletionParameters


class BedrockModelId(TypedDict):
    modelArn: str
    modelId: str


class IOutput(ABC):
    @abstractmethod
    def output(self) -> str:
        pass

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

    def output(self) -> str:
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

    def output(self) -> str:
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

    def output(self) -> str:
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


class TaggedAmazonResponse(AmazonResponse):
    provider: Literal["amazon"]


class TaggedAnthropicResponse(AnthropicResponse):
    provider: Literal["anthropic"]


class TaggedAI21Response(AI21Response):
    provider: Literal["ai21"]


class BedrockResponse(BaseModel, IOutput):
    __root__: Annotated[
        Union[
            TaggedAmazonResponse, TaggedAnthropicResponse, TaggedAI21Response
        ],
        Field(discriminator="provider"),
    ]

    def output(self) -> str:
        return self.__root__.output()

    def usage(self, prompt: str) -> TokenUsage:
        return self.__root__.usage(prompt)


class BedrockModels:
    def __init__(self, region: str):
        session = boto3.Session()

        self.bedrock = session.client(
            "bedrock",
            region,
            endpoint_url=f"https://bedrock.{region}.amazonaws.com",
        )

    def models(self) -> List[BedrockModelId]:
        return self.bedrock.list_foundation_models()["modelSummaries"]


log = logging.getLogger("bedrock")


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
    else:
        input_body["inputText"] = prompt

    return input_body


def prepare_model_kwargs(
    provider: str, model_params: CompletionParameters
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

        if model_params.top_k is not None:
            model_kwargs["top_k"] = model_params.top_k

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


class BedrockCustom(ChatModel):
    def __init__(
        self,
        model_id: str,
        region: str,
        model_params: CompletionParameters,
    ):
        self.model_id = model_id
        self.model_params = model_params

        provider = model_id.split(".")[0]

        self.model_kwargs = prepare_model_kwargs(provider, model_params)

        session = boto3.Session()
        self.bedrock = session.client(
            "bedrock",
            region,
            endpoint_url=f"https://bedrock.{region}.amazonaws.com",
        )

    def _call(self, prompt: str) -> Tuple[str, TokenUsage]:
        log.debug(f"prompt:\n{prompt}")

        provider = self.model_id.split(".")[0]

        model_response = self.bedrock.invoke_model(
            body=json.dumps(prepare_input(provider, prompt, self.model_kwargs)),
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
        )

        body = json.loads(model_response["body"].read())
        resp = BedrockResponse.parse_obj({"provider": provider, **body})
        response, usage = resp.output(), resp.usage(prompt)

        log.debug(f"response:\n{response}")
        return response, usage
