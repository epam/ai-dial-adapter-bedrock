from typing import Any

import requests

from tests.conftest import BASE_URL


def test_availability(server) -> Any:
    response = requests.get(f"{BASE_URL}/healthcheck")
    assert response.status_code == 200
