import anthropic
import httpx

from aidial_adapter_bedrock.server.exceptions import to_dial_exception


def test_pool_timeout_maps_to_503():
    request = httpx.Request("POST", "https://example.com")
    pool_timeout = httpx.PoolTimeout("pool exhausted", request=request)
    try:
        raise anthropic.APITimeoutError(request=request) from pool_timeout
    except anthropic.APITimeoutError as wrapped:
        anthropic_error = wrapped

    for e in [pool_timeout, anthropic_error]:
        assert to_dial_exception(e).status_code == 503


def test_other_timeout_is_not_503():
    request = httpx.Request("POST", "https://example.com")
    try:
        raise anthropic.APITimeoutError(request=request) from httpx.ReadTimeout(
            "read timeout", request=request
        )
    except anthropic.APITimeoutError as e:
        assert to_dial_exception(e).status_code == 500
