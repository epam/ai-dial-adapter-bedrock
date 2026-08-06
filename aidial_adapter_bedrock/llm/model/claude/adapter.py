import dataclasses
from typing import Literal, assert_never

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from aidial_adapter_anthropic.adapter.claude import ApproximateTokenizer
from aidial_adapter_anthropic.adapter.claude import (
    create_adapter as create_anthropic_adapter,
)
from typing_extensions import override

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from aidial_adapter_bedrock.deployments import ClaudeDeployment
from aidial_adapter_bedrock.dial_api.storage import create_file_storage
from aidial_adapter_bedrock.llm.model.conf import CLAUDE_DEFAULT_MAX_TOKENS
from aidial_adapter_bedrock.upstream_config import UpstreamConfig
from aidial_adapter_bedrock.utils.adapter_deployment import AdapterDeployment


def _supports_thinking(deployment: ClaudeDeployment) -> bool:
    return deployment in {
        D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        D.ANTHROPIC_CLAUDE_V4_OPUS,
        D.ANTHROPIC_CLAUDE_V4_1_OPUS,
        D.ANTHROPIC_CLAUDE_V4_6_OPUS,
        D.ANTHROPIC_CLAUDE_V4_7_OPUS,
        D.ANTHROPIC_CLAUDE_V4_SONNET,
        D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V4_5_SONNET,
        D.ANTHROPIC_CLAUDE_V4_6_SONNET,
        D.ANTHROPIC_CLAUDE_V4_8_OPUS,
    }


def _supports_documents(deployment: ClaudeDeployment) -> bool:
    return deployment in {
        D.ANTHROPIC_CLAUDE_V3_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
        D.ANTHROPIC_CLAUDE_V3_5_SONNET,
        D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        D.ANTHROPIC_CLAUDE_V4_OPUS,
        D.ANTHROPIC_CLAUDE_V4_6_OPUS,
        D.ANTHROPIC_CLAUDE_V4_7_OPUS,
        D.ANTHROPIC_CLAUDE_V4_SONNET,
        D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V4_5_SONNET,
        D.ANTHROPIC_CLAUDE_V4_6_SONNET,
        D.ANTHROPIC_CLAUDE_V4_8_OPUS,
    }


@dataclasses.dataclass
class _Tokenizer(ApproximateTokenizer):
    deployment: ClaudeDeployment

    @override
    def tokenize_tool_system_message(
        self, tool_choice: Literal["none", "auto", "any", "tool"]
    ) -> int:
        # See for the reference:
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview#pricing
        def _tokens() -> tuple[int, int]:
            match self.deployment:
                case D.ANTHROPIC_CLAUDE_V3_5_SONNET:
                    return (294, 261)
                case (
                    D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2
                    | D.ANTHROPIC_CLAUDE_V3_7_SONNET
                ):
                    return (346, 313)
                case (
                    D.ANTHROPIC_CLAUDE_V4_OPUS
                    | D.ANTHROPIC_CLAUDE_V4_1_OPUS
                    | D.ANTHROPIC_CLAUDE_V4_SONNET
                ):
                    return (313, 315)
                case (
                    D.ANTHROPIC_CLAUDE_V4_5_HAIKU
                    | D.ANTHROPIC_CLAUDE_V4_5_HAIKU_MANTLE
                    | D.ANTHROPIC_CLAUDE_V4_5_SONNET
                ):
                    return (496, 588)
                case (
                    D.ANTHROPIC_CLAUDE_V4_6_OPUS
                    | D.ANTHROPIC_CLAUDE_V4_6_SONNET
                ):
                    return (497, 589)
                case D.ANTHROPIC_CLAUDE_V5_SONNET:
                    return (354, 474)
                case D.ANTHROPIC_CLAUDE_V3_OPUS:
                    return (530, 281)
                case D.ANTHROPIC_CLAUDE_V3_SONNET:
                    return (159, 235)
                case D.ANTHROPIC_CLAUDE_V3_HAIKU:
                    return (264, 340)
                case D.ANTHROPIC_CLAUDE_V3_5_HAIKU:
                    return (264, 355)
                case D.ANTHROPIC_CLAUDE_V4_7_OPUS:
                    return (675, 804)
                case D.ANTHROPIC_CLAUDE_V4_8_OPUS:
                    return (290, 410)
                case D.ANTHROPIC_CLAUDE_V5_OPUS | D.ANTHROPIC_CLAUDE_V5_FABLE:
                    return (286, 406)
                case _:
                    assert_never(self.deployment)

        tokens1, tokens2 = _tokens()
        return tokens1 if tool_choice in ("auto", "none") else tokens2


async def create_adapter(
    deployment: AdapterDeployment[ClaudeDeployment],
    api_key: str,
    upstream_config: UpstreamConfig,
) -> ChatCompletionAdapter:
    ref = deployment.reference_deployment_id
    client = await create_anthropic_client(upstream_config)

    return await create_anthropic_adapter(
        deployment=deployment.upstream_deployment_id,
        storage=create_file_storage(api_key),
        client=client,
        custom_tokenizer=_Tokenizer(deployment.reference_deployment_id),
        default_max_tokens=CLAUDE_DEFAULT_MAX_TOKENS,
        supports_thinking=_supports_thinking(ref),
        supports_documents=_supports_documents(ref),
    )
