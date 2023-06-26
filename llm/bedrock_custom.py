import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Literal, Optional, Tuple, TypedDict, Union

import boto3
from anthropic.tokenizer import count_tokens
from langchain.llms.bedrock import LLMInputOutputAdapter
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llm.chat_model import ChatModel, TokenUsage


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
    def __init__(self, region: str = "us-east-1"):
        session = boto3.Session()

        self.bedrock = session.client(
            "bedrock",
            region,
            endpoint_url=f"https://bedrock.{region}.amazonaws.com",
        )

    def models(self) -> List[BedrockModelId]:
        return self.bedrock.list_foundation_models()["modelSummaries"]


log = logging.getLogger("bedrock")


class BedrockCustom(ChatModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: Optional[int],
        region: str = "us-east-1",
    ):
        self.model_id = model_id
        provider = model_id.split(".")[0]

        self.model_kwargs = {}
        if provider == "anthropic":
            self.model_kwargs["max_tokens_to_sample"] = (
                max_tokens if max_tokens is not None else 500
            )

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
            body=json.dumps(
                LLMInputOutputAdapter.prepare_input(
                    provider, prompt, self.model_kwargs
                )
            ),
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
        )

        body = json.loads(model_response["body"].read())
        resp = BedrockResponse.parse_obj({"provider": provider, **body})
        response, usage = resp.output(), resp.usage(prompt)

        log.debug(f"response:\n{response}")
        return response, usage
