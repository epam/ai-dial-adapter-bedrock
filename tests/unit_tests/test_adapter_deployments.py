import logging
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import pytest

from aidial_adapter_bedrock.adapter_deployments import AdapterDeployments
from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)


class Checker(Protocol):
    def check(self, deployments: AdapterDeployments): ...


@dataclass
class supported:
    deployment_id: ChatCompletionDeployment | EmbeddingsDeployment
    redirect: ChatCompletionDeployment | EmbeddingsDeployment | None = None

    def check(self, deployments: AdapterDeployments):
        deployment_name = self.deployment_id.value
        if isinstance(self.deployment_id, ChatCompletionDeployment):
            deployment = deployments.chat_completions.get(deployment_name)
        else:
            deployment = deployments.embeddings.get(deployment_name)

        assert deployment is not None
        assert deployment.adapter_deployment_id == deployment_name
        if self.redirect is not None:
            assert deployment.upstream_deployment_id == self.redirect.value
            assert deployment.reference_deployment_id == self.redirect
        else:
            assert deployment.upstream_deployment_id == deployment_name
            assert deployment.reference_deployment_id == self.deployment_id


@dataclass
class compat:
    deployment_id: str
    reference: ChatCompletionDeployment | EmbeddingsDeployment

    def check(self, deployments: AdapterDeployments):
        if isinstance(self.reference, ChatCompletionDeployment):
            deployment = deployments.chat_completions.get(self.deployment_id)
        else:
            deployment = deployments.embeddings.get(self.deployment_id)

        assert deployment is not None
        assert deployment.adapter_deployment_id == self.deployment_id
        assert deployment.upstream_deployment_id == self.deployment_id
        assert deployment.reference_deployment_id == self.reference


@dataclass
class TestCase:
    __test__ = False

    desc: str
    compat: Dict[str, str]

    error: str | None = None
    warning: str | None = None
    checks: List[Checker] = field(default_factory=list)


test_cases: List[TestCase] = [
    TestCase(
        desc="invalid compat",
        compat={"xxx": "yyy", "zzz": "ddd"},
        error='None of the values in the following compatibility mapping corresponds to a Bedrock deployment supported by the adapter: {"xxx": "yyy", "zzz": "ddd"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="partially invalid compat",
        compat={
            "xxx": "yyy",
            "zzz": ChatCompletionDeployment.AI21_J2_ULTRA_V1.value,
        },
        error='None of the values in the following compatibility mapping corresponds to a Bedrock deployment supported by the adapter: {"xxx": "yyy"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="compat chat+embeddings",
        compat={
            "xxx": ChatCompletionDeployment.AI21_J2_ULTRA_V1.value,
            "yyy": EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2.value,
        },
        checks=[
            supported(ChatCompletionDeployment.AI21_J2_ULTRA_V1),
            supported(EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2),
            supported(
                ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL,
                redirect=ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL_V1,
            ),
            compat("xxx", ChatCompletionDeployment.AI21_J2_ULTRA_V1),
            compat("yyy", EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2),
        ],
    ),
    TestCase(
        desc="compat supported deployment",
        compat={
            ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL.value: ChatCompletionDeployment.AI21_J2_ULTRA_V1.value,
        },
        checks=[
            supported(ChatCompletionDeployment.AI21_J2_ULTRA_V1),
            compat(
                ChatCompletionDeployment.STABILITY_STABLE_DIFFUSION_XL.value,
                ChatCompletionDeployment.AI21_J2_ULTRA_V1,
            ),
        ],
    ),
    TestCase(
        desc="compat mismatching supported deployments #1",
        compat={
            ChatCompletionDeployment.AI21_J2_ULTRA_V1.value: EmbeddingsDeployment.AMAZON_TITAN_EMBED_IMAGE_V1.value,
        },
        error="The chat completion deployment 'ai21.j2-ultra-v1' is mapped onto the embeddings deployment 'amazon.titan-embed-image-v1'",
    ),
    TestCase(
        desc="compat mismatching supported deployments #2",
        compat={
            EmbeddingsDeployment.AMAZON_TITAN_EMBED_IMAGE_V1.value: ChatCompletionDeployment.AI21_J2_ULTRA_V1.value,
        },
        error="The embeddings deployment 'amazon.titan-embed-image-v1' is mapped onto the chat completion deployment 'ai21.j2-ultra-v1'",
    ),
    TestCase(
        desc="outdated compatibility mapping (original)",
        compat={
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET.value: ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU.value
        },
        warning="'anthropic.claude-3-7-sonnet-20250219-v1:0' is one of the Bedrock deployments supported by the adapter already. Remove 'anthropic.claude-3-7-sonnet-20250219-v1:0' from the COMPATIBILITY_MAPPING variable to avoid the warning, otherwise you are losing the features present in the former deployment and missing from the latter.",
        checks=[
            supported(ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU),
            compat(
                ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET.value,
                ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU,
            ),
        ],
    ),
    TestCase(
        desc="outdated compatibility mapping (regional)",
        compat={
            ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET.US.value: ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU.value
        },
        warning="'us.anthropic.claude-3-7-sonnet-20250219-v1:0' is one of the Bedrock deployments supported by the adapter already. Remove 'us.anthropic.claude-3-7-sonnet-20250219-v1:0' from the COMPATIBILITY_MAPPING variable to avoid the warning, otherwise you are losing the features present in the former deployment and missing from the latter.",
        checks=[
            supported(ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU),
            compat(
                ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET.US.value,
                ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU,
            ),
        ],
    ),
]


@pytest.mark.parametrize(
    "test_case", test_cases, ids=lambda t: t.desc.replace(" ", "_")
)
def test_compat_mapping(caplog, test_case: TestCase):
    with caplog.at_level(logging.WARNING):
        if test_case.error is not None:
            with pytest.raises(ValueError, match=test_case.error):
                AdapterDeployments.create(compat_mapping=test_case.compat)
        else:
            deployments = AdapterDeployments.create(
                compat_mapping=test_case.compat
            )
            for checker in test_case.checks:
                checker.check(deployments)

    log_records = caplog.record_tuples

    if warn_message := test_case.warning:
        assert len(log_records) == 1
        _name, _level, message = log_records[0]
        assert message == warn_message
