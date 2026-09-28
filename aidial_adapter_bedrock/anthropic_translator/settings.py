import os


def get_api_version() -> str:
    return os.getenv("DIAL_API_VERSION") or "2025-01-01-preview"
