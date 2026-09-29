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
