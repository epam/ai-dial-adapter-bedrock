import os

import botocore.client
import httpx

from aidial_adapter_bedrock.utils.env import get_env_int

# The size of the thread pool running the blocking requests (e.g. AWS SDK calls)
THREAD_POOL_SIZE = get_env_int("THREAD_POOL_SIZE", 512)

# The maximum number of cached AWS clients. The cache is keyed by the upstream
# config and the session tags, so below the number of combinations in use
# every request re-creates a client and re-runs `assume_role`.
CLIENT_CACHE_MAX_SIZE = 512

ANTHROPIC_MAX_CONNECTIONS = get_env_int("ANTHROPIC_MAX_CONNECTIONS", 1000)
ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS = get_env_int(
    "ANTHROPIC_MAX_KEEPALIVE_CONNECTIONS", 100
)
ANTHROPIC_MAX_RETRY_ATTEMPTS = get_env_int("ANTHROPIC_MAX_RETRY_ATTEMPTS", 0)

BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS = get_env_int(
    "BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS", 1000
)

# Connect timeout for all upstream requests
CONNECT_TIMEOUT = 5
# Read timeout for long-running inference requests that can take minutes
READ_TIMEOUT = get_env_int("REQUEST_TIMEOUT_SECONDS", 10 * 60)
# A Botocore control-plane requests (e.g. STS, Creds refresh) answer in milliseconds; no need for long timeout.
_BOTOCORE_CONTROL_PLANE_READ_TIMEOUT = 10


def _get_generation_max_retry_attempts() -> int:
    if (value := os.getenv("BOTOCORE_MAX_RETRY_ATTEMPTS")) is not None:
        return int(value)
    if (value := os.getenv("AWS_MAX_ATTEMPTS")) is not None:
        return int(value) - 1
    return 0


# The config of the short-lived control-plane calls.
DEFAULT_BOTOCORE_CONFIG = botocore.client.Config(  # type: ignore
    # The max number of connections to the same upstream that are persisted
    # (saved to a connection pool). Greater number of connections *don't
    # block* each other.
    max_pool_connections=BOTOCORE_CLIENT_MAX_POOL_CONNECTIONS,
    connect_timeout=CONNECT_TIMEOUT,
    read_timeout=_BOTOCORE_CONTROL_PLANE_READ_TIMEOUT,
    retries={"mode": "standard", "total_max_attempts": 3},
)

# The config of the clients that generate: Bedrock Runtime and Converse.
GENERATION_CONFIG = DEFAULT_BOTOCORE_CONFIG.merge(
    botocore.client.Config(  # type: ignore
        read_timeout=READ_TIMEOUT,
        retries={
            "mode": "standard",
            "total_max_attempts": 1 + _get_generation_max_retry_attempts(),
        },
    )
)

# Same as Anthropic SDK timeouts: anthropic._constants.DEFAULT_TIMEOUT
DEFAULT_TIMEOUTS = httpx.Timeout(timeout=READ_TIMEOUT, connect=CONNECT_TIMEOUT)
