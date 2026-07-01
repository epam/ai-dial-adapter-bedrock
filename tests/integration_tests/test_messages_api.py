import json
from collections.abc import Awaitable, Callable
from typing import Any

import anthropic
import httpx
import pytest

from aidial_adapter_bedrock.deployments import ChatCompletionDeployment as D
from tests.integration_tests.test_chat_completion import (
    _DEPLOYMENT_TO_REGION,
    Deployment,
    DeploymentSpec,
    deployments,
    is_claude,
    select,
)
from tests.utils.openai import sanitize_test_name
from tests.utils.selector import pred

_claude_deployments = select(pred(is_claude), deployments)
_AWS_CLIENT_SWITCH_REGION = _DEPLOYMENT_TO_REGION[
    D.ANTHROPIC_CLAUDE_V4_7_OPUS.US
]


def _anthropic_text(response: anthropic.types.Message) -> str:
    return "".join(
        block.text
        for block in response.content
        if isinstance(block, anthropic.types.TextBlock)
    )


def _deployment_spec(deps: list[Deployment]):
    specs = [
        DeploymentSpec(
            region=_DEPLOYMENT_TO_REGION[dep],
            deployment=dep,
            optimized_latency=False,
        )
        for dep in deps
    ]
    return pytest.mark.parametrize(
        "deployment_spec",
        specs,
        ids=lambda x: sanitize_test_name(x.deployment.value),
    )


@pytest.fixture
def deployment(deployment_spec: DeploymentSpec) -> str:
    return deployment_spec.deployment.value


@pytest.fixture
def region(deployment_spec: DeploymentSpec) -> str:
    return deployment_spec.region


@pytest.fixture(params=[True, False], ids=["stream", "block"])
def stream(request) -> bool:
    return request.param


@pytest.fixture
async def anthropic_client(test_http_client: httpx.AsyncClient, region: str):
    async with anthropic.AsyncAnthropic(
        api_key="dummy-key",
        base_url="http://test-app.com/anthropic",
        http_client=test_http_client,
        max_retries=0,
        default_headers={
            "x-upstream-extra-data": json.dumps({"region": region})
        },
    ) as client:
        yield client


Messages = Callable[..., Awaitable[str]]


@pytest.fixture
def messages(
    deployment: str,
    anthropic_client: anthropic.AsyncAnthropic,
    stream: bool,
) -> Messages:
    async def _inner(msgs: list, max_tokens: int = 100, **kwargs: Any) -> str:
        if stream:
            async with anthropic_client.messages.stream(
                model=deployment,
                messages=msgs,
                max_tokens=max_tokens,
                **kwargs,
            ) as s:
                return await s.get_final_text()
        else:
            response = await anthropic_client.messages.create(
                model=deployment,
                messages=msgs,
                max_tokens=max_tokens,
                **kwargs,
            )
            return _anthropic_text(response)

    return _inner


@_deployment_spec(_claude_deployments)
async def test_2_plus_3(messages: Messages):
    text = await messages([{"role": "user", "content": "compute (2+3)"}])
    assert "5" in text


class TestUnknownDeployment:
    @pytest.fixture
    def deployment(self) -> str:
        return "unknown-deployment"

    @pytest.fixture
    def region(self) -> str:
        return "us-east-1"

    async def test_unknown_deployment(self, messages: Messages):
        with pytest.raises(
            anthropic.BadRequestError,
            match="The provided model identifier is invalid",
        ):
            await messages([{"role": "user", "content": "test"}], max_tokens=1)


@pytest.mark.parametrize(
    "client, model, default_client",
    [
        ("legacy", D.ANTHROPIC_CLAUDE_V4_7_OPUS.US.value, None),
        ("mantle", D.ANTHROPIC_CLAUDE_V4_7_OPUS.US.value, None),
        (None, D.ANTHROPIC_CLAUDE_V4_7_OPUS.US.value, "legacy"),
    ],
    ids=["legacy", "mantle", "default_from_env"],
)
async def test_aws_client_switch(
    test_http_client: httpx.AsyncClient,
    monkeypatch,
    client: str | None,
    model: str,
    default_client: str | None,
):
    if default_client is None:
        monkeypatch.delenv("AWS_CLAUDE_DEFAULT_CLIENT", raising=False)
    else:
        monkeypatch.setenv("AWS_CLAUDE_DEFAULT_CLIENT", default_client)

    extra_data: dict[str, str] = {"region": _AWS_CLIENT_SWITCH_REGION}
    if client is not None:
        extra_data["claude_client"] = client

    async with anthropic.AsyncAnthropic(
        api_key="dummy-key",
        base_url="http://test-app.com/anthropic",
        http_client=test_http_client,
        max_retries=0,
        default_headers={"x-upstream-extra-data": json.dumps(extra_data)},
    ) as anthropic_client:
        response = await anthropic_client.messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Compute 2+2"}],
        )

    assert "4" in _anthropic_text(response)
