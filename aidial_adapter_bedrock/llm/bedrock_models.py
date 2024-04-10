from enum import Enum


class BedrockDeployment(str, Enum):
    AMAZON_TITAN_TG1_LARGE = "amazon.titan-tg1-large"
    AI21_J2_GRANDE_INSTRUCT = "ai21.j2-grande-instruct"
    AI21_J2_JUMBO_INSTRUCT = "ai21.j2-jumbo-instruct"
    ANTHROPIC_CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1"
    ANTHROPIC_CLAUDE_V2 = "anthropic.claude-v2"
    ANTHROPIC_CLAUDE_V2_1 = "anthropic.claude-v2:1"
    ANTHROPIC_CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    ANTHROPIC_CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    STABILITY_STABLE_DIFFUSION_XL = "stability.stable-diffusion-xl"
    META_LLAMA2_13B_CHAT_V1 = "meta.llama2-13b-chat-v1"
    META_LLAMA2_70B_CHAT_V1 = "meta.llama2-70b-chat-v1"
    COHERE_COMMAND_TEXT_V14 = "cohere.command-text-v14"
    COHERE_COMMAND_LIGHT_TEXT_V14 = "cohere.command-light-text-v14"

    @property
    def deployment_id(self) -> str:
        """Deployment id under which the model is served by the adapter."""
        return self.value

    @property
    def model_id(self) -> str:
        """Id of the model in the Bedrock service."""
        return self.value

    @classmethod
    def from_deployment_id(cls, deployment_id: str) -> "BedrockDeployment":
        return cls(deployment_id)
