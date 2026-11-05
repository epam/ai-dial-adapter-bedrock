import json

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport
from openai import AsyncAzureOpenAI

from aidial_adapter_bedrock.utils.env import get_aws_default_region


@pytest.fixture(autouse=True)
def configure_unit_tests(monkeypatch, request):
    """
    Set up fake environment variables for unit tests.
    """
    if "tests/unit_tests" in request.node.nodeid:
        get_aws_default_region.cache_clear()
        monkeypatch.setenv("AWS_DEFAULT_REGION", "test-region")


@pytest.fixture
async def test_http_client():
    from aidial_adapter_bedrock.app import app

    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=ASGITransport(app),  # type: ignore
            base_url="http://test-app.com",
            params={"api-version": "dummy-version"},
            headers={"api-key": "dummy-key"},
        ) as client,
    ):
        yield client


def _get_extra_headers(region: str) -> dict[str, str]:
    return {"x-upstream-extra-data": json.dumps({"region": region})}


@pytest.fixture
def get_openai_client(test_http_client: httpx.AsyncClient):
    def _get_client(
        deployment_id: str | None = None,
        *,
        region: str | None = None,
        extra_headers: dict | None = None,
    ) -> AsyncAzureOpenAI:
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        if region:
            headers["region"] = region

        default_headers = {}
        if headers:
            default_headers["x-upstream-extra-data"] = json.dumps(headers)
        # DIAL Core sends the header, and the SDK requires it on chat completions
        if deployment_id:
            default_headers["X-DIAL-DEPLOYMENT-ID"] = deployment_id

        return AsyncAzureOpenAI(
            azure_endpoint=str(test_http_client.base_url),
            azure_deployment=deployment_id,
            api_version="dummy-version",
            api_key="dummy-key",
            max_retries=2,
            timeout=30,
            http_client=test_http_client,
            default_headers=default_headers,
        )

    yield _get_client
