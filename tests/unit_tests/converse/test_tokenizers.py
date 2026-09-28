from unittest.mock import MagicMock

import pytest

from aidial_adapter_bedrock.bedrock import Bedrock
from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as CCD
from aidial_adapter_bedrock.llm.converse.factory import _get_tokenizer_factory
from aidial_adapter_bedrock.llm.converse.tokenizers import (
    default_converse_tokenizer_factory,
    upstream_converse_tokenizer_factory,
)
from aidial_adapter_bedrock.llm.converse.types import (
    ConverseMessage,
    ConverseRequestWrapper,
    ConverseRole,
    ConverseTextPart,
)
from aidial_adapter_bedrock.utils.list_projection import ListProjection

# Models whose token counting is delegated to the Bedrock `CountTokens` API.
# Keep in sync with `_get_tokenizer_factory`.
_UPSTREAM_TOKENIZER_DEPLOYMENTS = {
    CCD.ANTHROPIC_CLAUDE_V4_5_HAIKU,
    CCD.ANTHROPIC_CLAUDE_V4_SONNET,
    CCD.ANTHROPIC_CLAUDE_V4_5_SONNET,
    CCD.ANTHROPIC_CLAUDE_V4_6_OPUS,
    CCD.ANTHROPIC_CLAUDE_V4_6_SONNET,
    CCD.ANTHROPIC_CLAUDE_V4_1_OPUS,
    CCD.ANTHROPIC_CLAUDE_V5_FABLE,
}

# Image models that are not supported by the Converse API tokenizer at all.
_STABILITY_DEPLOYMENTS = {
    CCD.STABILITY_STABLE_DIFFUSION_3_5_LARGE_V1,
    CCD.STABILITY_STABLE_IMAGE_CORE_V1_1,
    CCD.STABILITY_STABLE_IMAGE_ULTRA_V1,
    CCD.STABILITY_STABLE_IMAGE_ULTRA_V1_1,
}


def _make_params() -> ConverseRequestWrapper:
    return ConverseRequestWrapper(
        messages=ListProjection(
            lst=[
                (
                    ConverseMessage(
                        role=ConverseRole.USER,
                        content=[ConverseTextPart(text="Hello world!")],
                    ),
                    {0},
                )
            ]
        )
    )


def _make_bedrock(count_tokens_result: dict) -> Bedrock:
    client = MagicMock()
    client.count_tokens.return_value = count_tokens_result
    return Bedrock(client)


@pytest.mark.parametrize("deployment", list(CCD), ids=lambda d: d.value)
def test_tokenizer_factory_routing(deployment: CCD):
    if deployment in _STABILITY_DEPLOYMENTS:
        with pytest.raises(
            ValueError,
            match=(
                "Stability AI deployments are not supported by "
                "Converse API adapter."
            ),
        ):
            _get_tokenizer_factory(deployment)
    elif deployment in _UPSTREAM_TOKENIZER_DEPLOYMENTS:
        assert (
            _get_tokenizer_factory(deployment)
            is upstream_converse_tokenizer_factory
        )
    else:
        assert (
            _get_tokenizer_factory(deployment)
            is default_converse_tokenizer_factory
        )


async def test_upstream_tokenizer_delegates_to_count_tokens_api():
    bedrock = _make_bedrock({"inputTokens": 42})
    params = _make_params()

    tokenizer = upstream_converse_tokenizer_factory("model-id", bedrock, params)
    token_count = await tokenizer(params.messages.lst)

    assert token_count == 42
    bedrock.client.count_tokens.assert_called_once_with(
        modelId="model-id",
        input={"converse": params.to_request()},
    )


async def test_default_tokenizer_counts_offline():
    bedrock = _make_bedrock({"inputTokens": 42})
    params = _make_params()

    tokenizer = default_converse_tokenizer_factory("model-id", bedrock, params)
    token_count = await tokenizer(params.messages.lst)

    assert token_count > 0
    bedrock.client.count_tokens.assert_not_called()
