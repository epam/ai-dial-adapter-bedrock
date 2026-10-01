from aidial_adapter_bedrock.utils.boto import close_client, create_client

_CREDS = {
    "region_name": "us-east-1",
    "aws_access_key_id": "A" * 20,
    "aws_secret_access_key": "S" * 40,
}


def _open_pools(client) -> int:
    return len(client._endpoint.http_session._manager.pools)


def _open_a_pool(client) -> None:
    # Pools are lazy, so a client only holds one once it has served a request.
    client._endpoint.http_session._manager.connection_from_url(
        "https://bedrock-runtime.us-east-1.amazonaws.com"
    )


def test_clients_do_not_share_a_connection_pool():
    """The shared Session pools the service models, not the connections."""

    one, two = (create_client("bedrock-runtime", **_CREDS) for _ in range(2))

    assert one is not two
    assert one._endpoint.http_session is not two._endpoint.http_session
    # The parsed service model is what the shared Session deduplicates.
    assert (
        one.meta.service_model._service_description
        is two.meta.service_model._service_description
    )


async def test_close_client_releases_the_pool():
    client = create_client("bedrock-runtime", **_CREDS)
    _open_a_pool(client)
    assert _open_pools(client) == 1

    await close_client(client)

    assert _open_pools(client) == 0


async def test_close_client_is_repeatable():
    client = create_client("bedrock-runtime", **_CREDS)
    _open_a_pool(client)

    await close_client(client)
    await close_client(client)

    assert _open_pools(client) == 0


def test_botocore_own_clients_get_the_control_plane_timeouts():
    """
    A credential refresh calls STS through a client botocore builds itself,
    passing only a signature version -- and it does so while holding
    `RefreshableCredentials._refresh_lock`, which every signing thread blocks
    on in the mandatory refresh window. The session default is the only way to
    keep it off botocore's 60s defaults.
    """
    from botocore import UNSIGNED
    from botocore.client import Config
    from botocore.utils import create_nested_client

    from aidial_adapter_bedrock.utils.boto import _botocore_session
    from aidial_adapter_bedrock.utils.constants import (
        _BOTOCORE_CONTROL_PLANE_READ_TIMEOUT,
        CONNECT_TIMEOUT,
    )

    refresh_client = create_nested_client(
        _botocore_session,
        "sts",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )

    assert (
        refresh_client.meta.config.read_timeout
        == _BOTOCORE_CONTROL_PLANE_READ_TIMEOUT
    )
    assert refresh_client.meta.config.connect_timeout == CONNECT_TIMEOUT


def test_an_explicit_config_overrides_the_session_default():
    """`GENERATION_CONFIG`'s long read timeout must survive the merge."""

    from aidial_adapter_bedrock.utils.constants import (
        GENERATION_CONFIG,
        READ_TIMEOUT,
    )

    client = create_client(
        "bedrock-runtime", config=GENERATION_CONFIG, **_CREDS
    )
    assert client.meta.config.read_timeout == READ_TIMEOUT
