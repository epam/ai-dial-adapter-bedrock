import json
import os
from functools import cache
from typing import Literal, cast, get_args

from aidial_adapter_bedrock.utils.log_config import app_logger as log

AWSClaudeClient = Literal["legacy", "mantle", "converse"]


def get_env(name: str, err_msg: str | None = None) -> str:
    if (val := os.getenv(name)) is None:
        raise Exception(err_msg or f"{name} env variable is not set")
    return val


def get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


def get_env_bool(name: str, default: bool = False) -> bool:
    if (value := os.getenv(name)) is None:
        return default
    return value.strip().lower() in {"1", "true"}


def get_env_list(name: str) -> list[str] | None:
    if (value := os.getenv(name)) is not None:
        return [str.strip(s) for s in value.split(",")]
    return None


def get_str_dict(name: str) -> dict[str, str]:
    if (val := os.getenv(name)) is None:
        return {}

    try:
        dct = json.loads(val)
        assert isinstance(dct, dict)
        assert all(
            isinstance(k, str) and isinstance(v, str) for k, v in dct.items()
        )
        return dct
    except Exception:
        raise ValueError(
            f"{name} env variable doesn't contain a valid string to string JSON dictionary"
        ) from None


@cache
def get_aws_default_region() -> str:
    region = os.getenv("DEFAULT_REGION")
    if region is not None:
        log.warning(
            "DEFAULT_REGION env variable is deprecated. Use AWS_DEFAULT_REGION instead."
        )
        return region

    region = os.getenv("AWS_DEFAULT_REGION")
    if region is not None:
        return region

    raise ValueError("AWS_DEFAULT_REGION env variable is not set")


def get_default_claude_client() -> AWSClaudeClient:
    client = os.getenv("AWS_CLAUDE_DEFAULT_CLIENT", "legacy")
    allowed_clients = get_args(AWSClaudeClient)
    if client in allowed_clients:
        return cast(AWSClaudeClient, client)

    allowed_clients_str = ", ".join(
        f"'{allowed_client}'" for allowed_client in allowed_clients
    )
    raise ValueError(
        f"AWS_CLAUDE_DEFAULT_CLIENT env variable must be one of: {allowed_clients_str}"
    )
