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

    def check(self, deployments: AdapterDeployments):
        deployment_name = self.deployment_id.value
        if isinstance(self.deployment_id, ChatCompletionDeployment):
            deployment = deployments.chat_completions.get(deployment_name)
        else:
            deployment = deployments.embeddings.get(deployment_name)

        assert deployment is not None
        assert deployment.upstream_deployment_id == deployment_name
        assert deployment.compatible_deployment_id == self.deployment_id


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
        assert deployment.upstream_deployment_id == self.deployment_id
        assert deployment.compatible_deployment_id == self.reference


@dataclass
class TestCase:
    __test__ = False

    desc: str
    compat: Dict[str, str]

    error: str | None = None
    warning: str | None = None
    checks: List[Checker] = field(default_factory=list)


_CHAT_MODEL_1 = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_5_HAIKU
_CHAT_MODEL_2 = ChatCompletionDeployment.ANTHROPIC_CLAUDE_V3_7_SONNET

_EMBEDDING_MODEL = EmbeddingsDeployment.AMAZON_TITAN_EMBED_TEXT_V2

_outdated_mapping_warning_message = (
    "{deployment_id!r} deployment is already natively supported by the adapter, but it is also mapped to {supported_id!r} in the COMPATIBILITY_MAPPING variable. "
    "To avoid this warning and ensure you retain all features of {deployment_id!r}, remove it from the mapping. "
    "Otherwise, you may lose features that exist in {deployment_id!r} but are missing in {supported_id!r}."
)

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
            "zzz": _CHAT_MODEL_1.value,
        },
        error='None of the values in the following compatibility mapping corresponds to a Bedrock deployment supported by the adapter: {"xxx": "yyy"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="compat chat+embeddings",
        compat={
            "xxx": _CHAT_MODEL_1.value,
            "yyy": _EMBEDDING_MODEL.value,
        },
        checks=[
            supported(_CHAT_MODEL_1),
            supported(_EMBEDDING_MODEL),
            compat("xxx", _CHAT_MODEL_1),
            compat("yyy", _EMBEDDING_MODEL),
        ],
    ),
    TestCase(
        desc="compat mismatching supported deployments #1",
        compat={
            _CHAT_MODEL_1.value: _EMBEDDING_MODEL.value,
        },
        error=(
            f"The chat completion deployment {_CHAT_MODEL_1.value!r} is mapped"
            f" onto the embeddings deployment {_EMBEDDING_MODEL.value!r}"
        ),
    ),
    TestCase(
        desc="compat mismatching supported deployments #2",
        compat={
            _EMBEDDING_MODEL.value: _CHAT_MODEL_1.value,
        },
        error=(
            f"The embeddings deployment {_EMBEDDING_MODEL.value!r} is mapped"
            f" onto the chat completion deployment {_CHAT_MODEL_1.value!r}"
        ),
    ),
    TestCase(
        desc="outdated compatibility mapping (original)",
        compat={_CHAT_MODEL_2.value: _CHAT_MODEL_1.value},
        warning=_outdated_mapping_warning_message.format(
            deployment_id=_CHAT_MODEL_2.value,
            supported_id=_CHAT_MODEL_1.value,
        ),
        checks=[
            supported(_CHAT_MODEL_1),
            compat(_CHAT_MODEL_2.value, _CHAT_MODEL_1),
        ],
    ),
    TestCase(
        desc="outdated compatibility mapping (regional)",
        compat={_CHAT_MODEL_2.US.value: _CHAT_MODEL_1.value},
        warning=_outdated_mapping_warning_message.format(
            deployment_id=_CHAT_MODEL_2.US.value,
            supported_id=_CHAT_MODEL_1.value,
        ),
        checks=[
            supported(_CHAT_MODEL_1),
            compat(_CHAT_MODEL_2.US.value, _CHAT_MODEL_1),
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

    if warn_message := test_case.warning:
        assert warn_message in caplog.text
