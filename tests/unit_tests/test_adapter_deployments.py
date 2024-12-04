import operator
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import pytest

from aidial_adapter_bedrock.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_bedrock.utils.adapter_deployments import AdapterDeployments


class Checker(Protocol):
    def check(self, deployments: AdapterDeployments): ...


@dataclass
class supported:
    deployment_id: ChatCompletionDeployment | EmbeddingsDeployment
    redirect: ChatCompletionDeployment | EmbeddingsDeployment | None = None

    def check(self, deployments: AdapterDeployments):
        deployment_name = str(self.deployment_id)
        if isinstance(self.deployment_id, ChatCompletionDeployment):
            deployment = deployments.chat_completions.get(deployment_name)
        else:
            deployment = deployments.embeddings.get(deployment_name)

        assert deployment is not None
        assert deployment.adapter_deployment_id == deployment_name
        if self.redirect is not None:
            assert deployment.upstream_deployment_id == str(self.redirect)
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
    checks: List[Checker] = field(default_factory=list)


test_cases: List[TestCase] = [
    TestCase(
        desc="invalid compat",
        compat={"xxx": "yyy", "zzz": "ddd"},
        error='None of the values in the following compatibility dictionary corresponds to a Bedrock deployment supported by the adapter: {"xxx": "yyy", "zzz": "ddd"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="partially invalid compat",
        compat={
            "xxx": "yyy",
            "zzz": ChatCompletionDeployment.AI21_J2_ULTRA_V1.value,
        },
        error='None of the values in the following compatibility dictionary corresponds to a Bedrock deployment supported by the adapter: {"xxx": "yyy"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="default",
        compat={
            "xxx": ChatCompletionDeployment.AI21_J2_ULTRA_V1.value,
            "yyy": EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2,
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
]


@pytest.mark.parametrize(
    "test_case", test_cases, ids=operator.attrgetter("desc")
)
def test_compat_mapping_errors(test_case: TestCase):
    if test_case.error is not None:
        with pytest.raises(ValueError, match=test_case.error):
            AdapterDeployments.create(compat_mapping=test_case.compat)
    else:
        deployments = AdapterDeployments.create(compat_mapping=test_case.compat)
        for checker in test_case.checks:
            checker.check(deployments)
