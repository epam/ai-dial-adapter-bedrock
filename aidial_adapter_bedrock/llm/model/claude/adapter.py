from aidial_adapter_anthropic.llm.chat_model import ChatCompletionAdapter
from aidial_adapter_anthropic.llm.model.claude.adapter import (
    create_adapter as create_anthropic_adapter,
)

from aidial_adapter_bedrock.bedrock import create_anthropic_client
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from aidial_adapter_bedrock.deployments import ClaudeDeployment
from aidial_adapter_bedrock.upstream_config import UpstreamConfig
from aidial_adapter_bedrock.utils.adapter_deployment import AdapterDeployment


def _supports_thinking(deployment: ClaudeDeployment) -> bool:
    return deployment in {
        D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        D.ANTHROPIC_CLAUDE_V4_OPUS,
        D.ANTHROPIC_CLAUDE_V4_1_OPUS,
        D.ANTHROPIC_CLAUDE_V4_SONNET,
        D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V4_5_SONNET,
    }


def _supports_documents(deployment: ClaudeDeployment) -> bool:
    return deployment in {
        D.ANTHROPIC_CLAUDE_V3_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V3_5_SONNET_V2,
        D.ANTHROPIC_CLAUDE_V3_5_SONNET,
        D.ANTHROPIC_CLAUDE_V3_7_SONNET,
        D.ANTHROPIC_CLAUDE_V4_OPUS,
        D.ANTHROPIC_CLAUDE_V4_SONNET,
        D.ANTHROPIC_CLAUDE_V4_5_HAIKU,
        D.ANTHROPIC_CLAUDE_V4_5_SONNET,
    }


async def create_adapter(
    deployment: AdapterDeployment[ClaudeDeployment],
    api_key: str,
    upstream_config: UpstreamConfig,
) -> ChatCompletionAdapter:
    ref = deployment.reference_deployment_id
    client = await create_anthropic_client(upstream_config)

    return await create_anthropic_adapter(
        deployment.upstream_deployment_id,
        api_key,
        client,
        supports_thinking=_supports_thinking(ref),
        supports_documents=_supports_documents(ref),
    )
