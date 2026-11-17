from enum import Enum
from typing import Literal, Self

from aidial_adapter_bedrock.utils.region_deployment import (
    RegionInferenceDeployment,
)


class ChatCompletionDeployment(RegionInferenceDeployment):
    AMAZON_NOVA_PRO = "amazon.nova-pro-v1:0"
    AMAZON_NOVA_LITE = "amazon.nova-lite-v1:0"
    AMAZON_NOVA_MICRO = "amazon.nova-micro-v1:0"

    AI21_JAMBA_1_5_LARGE_V1 = "ai21.jamba-1-5-large-v1:0"
    AI21_JAMBA_1_5_MINI_V1 = "ai21.jamba-1-5-mini-v1:0"

    ANTHROPIC_CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    ANTHROPIC_CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ANTHROPIC_CLAUDE_V3_5_SONNET_V2 = (
        "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    ANTHROPIC_CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    ANTHROPIC_CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    ANTHROPIC_CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    ANTHROPIC_CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    ANTHROPIC_CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    ANTHROPIC_CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"
    ANTHROPIC_CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    ANTHROPIC_CLAUDE_V4_5_HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE = "anthropic.claude-haiku-4-5"
    ANTHROPIC_CLAUDE_V4_5_SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    ANTHROPIC_CLAUDE_V4_6_OPUS = "anthropic.claude-opus-4-6-v1"
    ANTHROPIC_CLAUDE_V4_6_SONNET = "anthropic.claude-sonnet-4-6"
    ANTHROPIC_CLAUDE_V5_SONNET = "anthropic.claude-sonnet-5"
    ANTHROPIC_CLAUDE_V4_7_OPUS = "anthropic.claude-opus-4-7"
    ANTHROPIC_CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
    ANTHROPIC_CLAUDE_V5_OPUS = "anthropic.claude-opus-5"
    ANTHROPIC_CLAUDE_V5_FABLE = "anthropic.claude-fable-5"
    ANTHROPIC_CLAUDE_V5_5_HAIKU = "anthropic.claude-haiku-5-5"
    ANTHROPIC_CLAUDE_V5_5_OPUS = "anthropic.claude-opus-5-5"
    ANTHROPIC_CLAUDE_V5_5_SONNET = "anthropic.claude-sonnet-5-5"

    STABILITY_STABLE_DIFFUSION_3_5_LARGE_V1 = "stability.sd3-5-large-v1:0"
    STABILITY_STABLE_IMAGE_CORE_V1_1 = "stability.stable-image-core-v1:1"
    STABILITY_STABLE_IMAGE_ULTRA_V1 = "stability.stable-image-ultra-v1:0"
    STABILITY_STABLE_IMAGE_ULTRA_V1_1 = "stability.stable-image-ultra-v1:1"

    META_LLAMA3_8B_INSTRUCT_V1 = "meta.llama3-8b-instruct-v1:0"
    META_LLAMA3_70B_INSTRUCT_V1 = "meta.llama3-70b-instruct-v1:0"
    META_LLAMA3_1_8B_INSTRUCT_V1 = "meta.llama3-1-8b-instruct-v1:0"
    META_LLAMA3_1_70B_INSTRUCT_V1 = "meta.llama3-1-70b-instruct-v1:0"
    META_LLAMA3_1_405B_INSTRUCT_V1 = "meta.llama3-1-405b-instruct-v1:0"
    META_LLAMA3_2_1B_INSTRUCT_V1 = "meta.llama3-2-1b-instruct-v1:0"
    META_LLAMA3_2_3B_INSTRUCT_V1 = "meta.llama3-2-3b-instruct-v1:0"
    META_LLAMA3_2_11B_INSTRUCT_V1 = "meta.llama3-2-11b-instruct-v1:0"
    META_LLAMA3_2_90B_INSTRUCT_V1 = "meta.llama3-2-90b-instruct-v1:0"
    META_LLAMA3_3_70B_INSTRUCT_V1 = "meta.llama3-3-70b-instruct-v1:0"
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1 = (
        "meta.llama4-maverick-17b-instruct-v1:0"
    )
    META_LLAMA4_SCOUT_17B_INSTRUCT_V1 = "meta.llama4-scout-17b-instruct-v1:0"

    COHERE_COMMAND_R_V1 = "cohere.command-r-v1:0"
    COHERE_COMMAND_R_PLUS_V1 = "cohere.command-r-plus-v1:0"

    DEEPSEEK_R1_V2 = "deepseek.r1-v1:0"
    DEEPSEEK_V3_2 = "deepseek.v3.2"

    MINIMAX_M21 = "minimax.minimax-m2.1"
    MINIMAX_M25 = "minimax.minimax-m2.5"

    MISTRAL_DEVSTRAL_2_123B = "mistral.devstral-2-123b"

    MOONSHOTAI_KIMI_K2_5 = "moonshotai.kimi-k2.5"

    NVIDIA_NEMOTRON_NANO_9B_V2 = "nvidia.nemotron-nano-9b-v2"
    NVIDIA_NEMOTRON_NANO_12B_V2 = "nvidia.nemotron-nano-12b-v2"
    NVIDIA_NEMOTRON_NANO_3_30B = "nvidia.nemotron-nano-3-30b"
    NVIDIA_NEMOTRON_SUPER_3_120B = "nvidia.nemotron-super-3-120b"

    OPENAI_GPT_OSS_20B = "openai.gpt-oss-20b-1:0"
    OPENAI_GPT_OSS_120B = "openai.gpt-oss-120b-1:0"
    OPENAI_GPT_OSS_SAFEGUARD_20B = "openai.gpt-oss-safeguard-20b"
    OPENAI_GPT_OSS_SAFEGUARD_120B = "openai.gpt-oss-safeguard-120b"

    ZAI_GLM_4_7 = "zai.glm-4.7"
    ZAI_GLM_4_7_FLASH = "zai.glm-4.7-flash"
    ZAI_GLM_5 = "zai.glm-5"


ClaudeDeployment = Literal[
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_HAIKU,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_1_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_6_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_5_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_6_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V5_SONNET,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_5_HAIKU,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_7_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V4_8_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V5_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V5_FABLE,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V5_5_HAIKU,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V5_5_OPUS,
    ChatCompletionDeployment.ANTHROPIC_CLAUDE_V5_5_SONNET,
]


class EmbeddingsDeployment(Enum):
    AMAZON_TITAN_EMBED_TEXT_V1 = "amazon.titan-embed-text-v1"
    AMAZON_TITAN_EMBED_TEXT_V2 = "amazon.titan-embed-text-v2:0"
    AMAZON_TITAN_EMBED_IMAGE_V1 = "amazon.titan-embed-image-v1"

    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"

    @classmethod
    def deployments(cls) -> list[str]:
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, model_id: str) -> Self | None:
        for deployment in cls:
            if deployment.value == model_id:
                return deployment
        return None
