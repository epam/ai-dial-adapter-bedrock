import os
from typing import Optional

from aidial_adapter_bedrock.utils.log_config import app_logger as log


def get_env(name: str, err_msg: Optional[str] = None) -> str:
    if name in os.environ:
        val = os.environ.get(name)
        if val is not None:
            return val

    raise Exception(err_msg or f"{name} env variable is not set")


def get_env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() == "true"


def get_aws_default_region() -> str:
    region = os.getenv("AWS_DEFAULT_REGION")

    if region is None:
        region = os.getenv("DEFAULT_REGION")
        if region is None:
            raise ValueError("AWS_DEFAULT_REGION env variable is not set")

        log.warn(
            "DEFAULT_REGION is deprecated, use AWS_DEFAULT_REGION instead",
        )
        return region

    return region
