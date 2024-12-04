import operator
from dataclasses import dataclass
from typing import Dict, List

import pytest

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment
from aidial_adapter_bedrock.utils.adapter_deployments import AdapterDeployments


@dataclass
class TestCase:
    __test__ = False

    desc: str
    compat: Dict[str, str]

    error: str | None = None


test_cases: List[TestCase] = [
    TestCase(
        desc="invalid compat",
        compat={"xxx": "yyy", "zzz": "ddd"},
        error='None of the values in the following compatibility mapping maps to a Bedrock deployment supported by the adapter: {"xxx": "yyy", "zzz": "ddd"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
    TestCase(
        desc="partially invalid compat",
        compat={
            "xxx": "yyy",
            "zzz": ChatCompletionDeployment.AI21_J2_ULTRA_V1.value,
        },
        error='None of the values in the following compatibility mapping maps to a Bedrock deployment supported by the adapter: {"xxx": "yyy"}. Remap the deployments to the supported Bedrock deployments to fix the error.',
    ),
]


@pytest.mark.parametrize(
    "test_case", test_cases, ids=operator.attrgetter("desc")
)
def test_compat_mapping(test_case: TestCase):
    if test_case.error is not None:
        with pytest.raises(ValueError, match=test_case.error):
            AdapterDeployments.create(compat_mapping=test_case.compat)
