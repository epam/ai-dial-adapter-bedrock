from typing import Any

import requests

from tests.conftest import TEST_SERVER_URL


def test_availability(server) -> Any:
    response = requests.get(f"{TEST_SERVER_URL}/healthcheck")
    assert response.status_code == 200
