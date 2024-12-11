from typing import Any

import httpx


async def test_availability(test_http_client: httpx.AsyncClient) -> Any:
    response = await test_http_client.get("health")
    assert response.status_code == 200
