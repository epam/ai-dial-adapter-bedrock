from enum import Enum
from typing import Literal


class ChatCompletionDeployment(str, Enum):
    AMAZON_TITAN_TG1_LARGE = "amazon.titan-tg1-large"

    AI21_J2_GRANDE_INSTRUCT = "ai21.j2-grande-instruct"
    AI21_J2_JUMBO_INSTRUCT = "ai21.j2-jumbo-instruct"

    ANTHROPIC_CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1"
    ANTHROPIC_CLAUDE_V2 = "anthropic.claude-v2"
    ANTHROPIC_CLAUDE_V2_1 = "anthropic.claude-v2:1"

    ANTHROPIC_CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    ANTHROPIC_CLAUDE_V3_SONNET_US = "us.anthropic.claude-3-sonnet-20240229-v1:0"
    ANTHROPIC_CLAUDE_V3_SONNET_EU = "eu.anthropic.claude-3-sonnet-20240229-v1:0"
    ANTHROPIC_CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ANTHROPIC_CLAUDE_V3_5_SONNET_US = (
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    ANTHROPIC_CLAUDE_V3_5_SONNET_EU = (
        "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    ANTHROPIC_CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    ANTHROPIC_CLAUDE_V3_HAIKU_US = "us.anthropic.claude-3-haiku-20240307-v1:0"
    ANTHROPIC_CLAUDE_V3_HAIKU_EU = "eu.anthropic.claude-3-haiku-20240307-v1:0"
    ANTHROPIC_CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    ANTHROPIC_CLAUDE_V3_OPUS_US = "us.anthropic.claude-3-opus-20240229-v1:0"

    STABILITY_STABLE_DIFFUSION_XL = "stability.stable-diffusion-xl"
    STABILITY_STABLE_DIFFUSION_XL_V1 = "stability.stable-diffusion-xl-v1"

    META_LLAMA2_13B_CHAT_V1 = "meta.llama2-13b-chat-v1"
    META_LLAMA2_70B_CHAT_V1 = "meta.llama2-70b-chat-v1"
    META_LLAMA3_8B_INSTRUCT_V1 = "meta.llama3-8b-instruct-v1:0"
    META_LLAMA3_70B_INSTRUCT_V1 = "meta.llama3-70b-instruct-v1:0"
    META_LLAMA3_1_405B_INSTRUCT_V1 = "meta.llama3-1-405b-instruct-v1:0"
    META_LLAMA3_1_70B_INSTRUCT_V1 = "meta.llama3-1-70b-instruct-v1:0"
    META_LLAMA3_1_8B_INSTRUCT_V1 = "meta.llama3-1-8b-instruct-v1:0"

    COHERE_COMMAND_TEXT_V14 = "cohere.command-text-v14"
    COHERE_COMMAND_LIGHT_TEXT_V14 = "cohere.command-light-text-v14"

    @property
    def deployment_id(self) -> str:
        """Deployment id under which the model is served by the adapter."""
        return self.value

    @property
    def model_id(self) -> str:
        """Id of the model in the Bedrock service."""

        # Redirect Stability model without version to the earliest non-deprecated version (V1)
        if self == ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL:
            return (
                ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL_V1.model_id
            )

        return self.value

    @classmethod
    def from_deployment_id(
        cls, deployment_id: str
    ) -> "ChatCompletionDeployment":
        return cls(deployment_id)


Claude3Deployment = Literal[
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS,
]


class EmbeddingsDeployment(str, Enum):
    AMAZON_TITAN_EMBED_TEXT_V1 = "amazon.titan-embed-text-v1"
    AMAZON_TITAN_EMBED_TEXT_V2 = "amazon.titan-embed-text-v2:0"
    AMAZON_TITAN_EMBED_IMAGE_V1 = "amazon.titan-embed-image-v1"

    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"

    @property
    def deployment_id(self) -> str:
        """Deployment id under which the model is served by the adapter."""
        return self.value

    @property
    def model_id(self) -> str:
        """Id of the model in the Bedrock service."""
        return self.value
