import json
from typing import Mapping

import httpx
import pytest
from httpx import ASGITransport
from openai import AsyncAzureOpenAI


@pytest.fixture(autouse=True)
def configure_unit_tests(monkeypatch, request):
    """
    Set up fake environment variables for unit tests.
    """
    if "tests/unit_tests" in request.node.nodeid:
        monkeypatch.setenv("AWS_DEFAULT_REGION", "test-region")


@pytest.fixture
async def test_http_client():
    from aidial_adapter_bedrock.app import app

    async with httpx.AsyncClient(
        transport=ASGITransport(app),  # type: ignore
        base_url="http://test-app.com",
        params={"api-version": "dummy-version"},
        headers={"api-key": "dummy-key"},
    ) as client:
        yield client


def _get_extra_headers(region: str) -> Mapping[str, str]:
    return {"x-upstream-extra-data": json.dumps({"region": region})}


@pytest.fixture
def get_openai_client(test_http_client: httpx.AsyncClient):
    def _get_client(
        deployment_id: str | None = None,
        *,
        region: str | None = None,
    ) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=str(test_http_client.base_url),
            azure_deployment=deployment_id,
            api_version="dummy-version",
            api_key="dummy-key",
            max_retries=2,
            timeout=30,
            http_client=test_http_client,
            default_headers=_get_extra_headers(region) if region else {},
        )

    yield _get_client
