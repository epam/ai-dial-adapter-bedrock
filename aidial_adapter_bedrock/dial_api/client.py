import os

from aidial_client import AsyncDial

from aidial_adapter_bedrock.bedrock import get_dial_client_pool

DIAL_URL = os.getenv("DIAL_URL")


def create_dial_client(api_key: str) -> AsyncDial | None:
    if DIAL_URL is None:
        return None

    return get_dial_client_pool().create_client(
        base_url=DIAL_URL,
        api_key=api_key,
    )
