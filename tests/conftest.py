import os

import pytest

from tests.utils.server import server_generator

DEFAULT_API_VERSION = "2023-03-15-preview"
TEST_SERVER_URL = os.getenv("TEST_SERVER_URL", "http://0.0.0.0:5001")


@pytest.fixture(scope="module")
def server():
    yield from server_generator(
        "aidial_adapter_bedrock.app:app", TEST_SERVER_URL
    )
