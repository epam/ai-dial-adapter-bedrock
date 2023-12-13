import re
from enum import Enum
from typing import Dict


def _sanitize_deployment_name(id: str) -> str:
    """Make the deployment name match a pattern which is expected by the core.
    Replace non-compliant symbols with a dash.
    """
    return re.sub(r"[^-.@a-zA-Z0-9]", "-", id)


class BedrockDeployment(str, Enum):
    AMAZON_TITAN_TG1_LARGE = "amazon.titan-tg1-large"
    AI21_J2_GRANDE_INSTRUCT = "ai21.j2-grande-instruct"
    AI21_J2_JUMBO_INSTRUCT = "ai21.j2-jumbo-instruct"
    ANTHROPIC_CLAUDE_INSTANT_V1 = "anthropic.claude-instant-v1"
    ANTHROPIC_CLAUDE_V1 = "anthropic.claude-v1"
    ANTHROPIC_CLAUDE_V2 = "anthropic.claude-v2"
    ANTHROPIC_CLAUDE_V2_1 = "anthropic.claude-v2:1"
    STABILITY_STABLE_DIFFUSION_XL = "stability.stable-diffusion-xl"
    META_LLAMA2_13B_CHAT_V1 = "meta.llama2-13b-chat-v1"
    META_LLAMA2_70B_CHAT_V1 = "meta.llama2-70b-chat-v1"
    COHERE_COMMAND_TEXT_V14 = "cohere.command-text-v14"
    COHERE_COMMAND_LIGHT_TEXT_V14 = "cohere.command-light-text-v14"

    def get_deployment_id(self) -> str:
        return _sanitize_deployment_name(self.value)

    @classmethod
    def from_deployment_id(cls, deployment_id: str) -> "BedrockDeployment":
        model_id = _deployment_id_to_model_id.get(deployment_id)
        if model_id is None:
            raise ValueError(f"Unknown deployment: {deployment_id}")
        return cls(model_id)


def _build_reverse_mapping() -> Dict[str, str]:
    mapping = {}
    for deployment in BedrockDeployment:
        deployment_id = deployment.get_deployment_id()
        model_id = deployment.value
        if deployment_id in mapping:
            raise ValueError(
                f"The same deployment id '{deployment_id}' corresponds "
                f"to two model ids: '{model_id}' and '{mapping[deployment_id]}'"
            )
        mapping[deployment_id] = model_id
    return mapping


_deployment_id_to_model_id: Dict[str, str] = _build_reverse_mapping()
