import os

from aidial_adapter_anthropic.dial.storage import FileStorage

DIAL_URL = os.getenv("DIAL_URL")


def create_file_storage(api_key: str) -> FileStorage | None:
    if DIAL_URL is None:
        return None

    return FileStorage(dial_url=DIAL_URL, api_key=api_key)
