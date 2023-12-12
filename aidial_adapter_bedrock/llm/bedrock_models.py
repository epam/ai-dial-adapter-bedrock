from enum import Enum


class BedrockDeployment(str, Enum):
    AMAZON_TITAN_TG1_LARGE = "amazon.titan-tg1-large"
    AI21_J2_GRANDE_INSTRUCT = "ai21.j2-grande-instruct"
    AI21_J2_JUMBO_INSTRUCT = "ai21.j2-jumbo-instruct"
    AI21_J2_MID = "ai21.j2-mid"
    AI21_J2_ULTRA = "ai21.j2-ultra"
    ANTHROPIC_CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1"
    ANTHROPIC_CLAUDE_V1 = "anthropic.claude-v1"
    ANTHROPIC_CLAUDE_V2 = "anthropic.claude-v2"
    ANTHROPIC_CLAUDE_V2_1_200K = (
        "anthropic.claude-v2:1"  # "anthropic.claude-v2:1:200k"
    )
    STABILITY_STABLE_DIFFUSION_XL = "stability.stable-diffusion-xl"
    META_LLAMA2_13B_CHAT_V1 = "meta.llama2-13b-chat-v1"
    META_LLAMA2_70B_CHAT_V1 = "meta.llama2-70b-chat-v1"
    COHERE_COMMAND_TEXT_V14 = "cohere.command-text-v14"
    COHERE_COMMAND_LIGHT_TEXT_V14 = "cohere.command-light-text-v14"

    def get_model_id(self) -> str:
        return self.value
