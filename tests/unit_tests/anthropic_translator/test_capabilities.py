import httpx
import pytest
import respx

from aidial_adapter_bedrock.anthropic_translator.capabilities import (
    UNRESOLVED_PROFILE,
    NestedBudget,
    NestedEffort,
    TopLevelEffort,
    clear_cache,
    get_deployment_profile,
)
from tests.unit_tests.anthropic_translator.helpers import catalog

_CORE = "http://dial-core"
_MODELS = "/openai/models"
_KEY = ("api-key", "caller-key")


@pytest.fixture(autouse=True)
def _isolated_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def mock_core():
    with respx.mock(base_url=_CORE) as mock:
        yield mock


async def profile_for(body: dict, deployment: str = "gpt-5.5"):
    """One independent lookup against a catalog serving `body`."""
    clear_cache()
    with respx.mock(base_url=_CORE) as mock:
        mock.get(_MODELS).respond(json=body)
        return await get_deployment_profile(_CORE, _KEY, deployment)


async def test_features_are_read_from_the_catalog():
    profile = await profile_for(
        catalog(
            features={
                "temperature": True,
                "cache": True,
                "max_completion_tokens_supported": True,
            },
            limits={"max_completion_tokens": 4096},
        )
    )
    assert profile.temperature_supported is True
    assert profile.cache_supported is True
    assert profile.max_completion_tokens_supported is True
    assert profile.max_output_tokens == 4096


async def test_temperature_is_dropped_only_on_an_explicit_false():
    suppressed = await profile_for(catalog(features={"temperature": False}))
    assert suppressed.temperature_supported is False

    # Absent means unknown, and dropping it on a guess silently changes
    # generation quality.
    unknown = await profile_for({"object": "list", "data": [{"id": "gpt-5.5"}]})
    assert unknown.temperature_supported is True
    assert UNRESOLVED_PROFILE.temperature_supported is True


@pytest.mark.parametrize("cache", [False, None])
async def test_cache_stays_off_unless_the_catalog_turns_it_on(cache):
    profile = await profile_for(catalog(features={"cache": cache}))
    assert profile.cache_supported is False
    assert UNRESOLVED_PROFILE.cache_supported is False


async def test_max_completion_tokens_defaults_to_the_older_spelling():
    profile = await profile_for(catalog())
    assert profile.max_completion_tokens_supported is False
    assert UNRESOLVED_PROFILE.max_completion_tokens_supported is False


@pytest.mark.parametrize(
    "limits, defaults, expected",
    [
        # Every row of the token-budget table names `max_completion_tokens`
        # when present, and `defaults.max_tokens` otherwise.
        ({"max_total_tokens": 1000}, {"max_tokens": 64}, 64),
        ({"max_total_tokens": 1000, "max_completion_tokens": 128}, None, 128),
        ({"max_prompt_tokens": 900, "max_completion_tokens": 128}, None, 128),
        ({"max_prompt_tokens": 900}, {"max_tokens": 64}, 64),
        ({"max_completion_tokens": 128}, {"max_tokens": 64}, 128),
        (None, None, None),
    ],
)
async def test_output_ceiling_derivation(limits, defaults, expected):
    profile = await profile_for(catalog(limits=limits, defaults=defaults))
    assert profile.max_output_tokens == expected


async def test_reasoning_knob_prefers_nested_configuration():
    profile = await profile_for(
        catalog(
            features={"reasoning_efforts": ["low", "high"]},
            defaults={
                "custom_fields": {
                    "configuration": {"reasoning": {"summary": "auto"}}
                }
            },
        )
    )
    assert profile.reasoning == NestedEffort(defaults={"summary": "auto"})


async def test_thinking_key_selects_the_budget_knob():
    profile = await profile_for(
        catalog(
            defaults={
                "custom_fields": {
                    "configuration": {
                        "thinking": {
                            "include_thoughts": True,
                            "thinking_budget": 2048,
                        }
                    }
                }
            },
        )
    )
    assert profile.reasoning == NestedBudget(
        defaults={"include_thoughts": True, "thinking_budget": 2048}
    )


async def test_no_nested_configuration_falls_back_to_advertised_levels():
    profile = await profile_for(
        catalog(features={"reasoning_efforts": ["low", "high"]})
    )
    assert profile.reasoning == TopLevelEffort(levels=["low", "high"])


async def test_empty_reasoning_efforts_means_no_reasoning():
    profile = await profile_for(catalog(features={"reasoning_efforts": []}))
    assert profile.reasoning == TopLevelEffort(levels=[])
    assert UNRESOLVED_PROFILE.reasoning == TopLevelEffort(levels=[])


@pytest.mark.parametrize(
    "defaults",
    [
        "not-an-object",
        {"custom_fields": "not-an-object"},
        {"custom_fields": {"configuration": 42}},
        {"custom_fields": {"configuration": {"reasoning": "not-an-object"}}},
    ],
)
async def test_operator_supplied_defaults_degrade_instead_of_faulting(defaults):
    # `defaults` is copied verbatim from deployment configuration with no
    # schema validation, so any level of it can be anything.
    profile = await profile_for(catalog(defaults=defaults))
    assert profile.max_output_tokens is None


async def test_deployment_missing_from_the_catalog_is_unresolved():
    profile = await profile_for(catalog("other-model"), deployment="gpt-5.5")
    assert profile == UNRESOLVED_PROFILE


@pytest.mark.parametrize(
    "failure",
    [
        {"side_effect": httpx.ConnectError("refused")},
        {"side_effect": httpx.ReadTimeout("too slow")},
        {"return_value": httpx.Response(500)},
        {"return_value": httpx.Response(200, text="{not json")},
        {"return_value": httpx.Response(200, json={"data": "not-a-list"})},
    ],
)
async def test_a_failed_lookup_never_fails_the_request(
    mock_core: respx.MockRouter, failure
):
    mock_core.get(_MODELS).mock(**failure)
    profile = await get_deployment_profile(_CORE, _KEY, "gpt-5.5")
    assert profile == UNRESOLVED_PROFILE


async def test_failures_are_not_cached(mock_core: respx.MockRouter):
    route = mock_core.get(_MODELS).mock(
        side_effect=httpx.ConnectError("refused")
    )
    await get_deployment_profile(_CORE, _KEY, "gpt-5.5")
    await get_deployment_profile(_CORE, _KEY, "gpt-5.5")
    assert route.call_count == 2


async def test_a_successful_catalog_is_fetched_once(
    mock_core: respx.MockRouter,
):
    route = mock_core.get(_MODELS).respond(json=catalog())
    await get_deployment_profile(_CORE, _KEY, "gpt-5.5")
    await get_deployment_profile(_CORE, _KEY, "gpt-5.5")
    assert route.call_count == 1


async def test_each_credential_gets_its_own_catalog(
    mock_core: respx.MockRouter,
):
    # Core filters the listing by the caller's roles, so a narrow credential
    # must not answer for a broader one.
    route = mock_core.get(_MODELS).respond(json=catalog())
    await get_deployment_profile(_CORE, ("api-key", "one"), "gpt-5.5")
    await get_deployment_profile(_CORE, ("api-key", "two"), "gpt-5.5")
    assert route.call_count == 2


async def test_the_callers_credential_is_forwarded(
    mock_core: respx.MockRouter,
):
    route = mock_core.get(_MODELS).respond(json=catalog())
    await get_deployment_profile(_CORE, ("Authorization", "Bearer t"), "x")
    assert route.calls.last.request.headers["authorization"] == "Bearer t"


async def test_zero_ttl_disables_caching(
    mock_core: respx.MockRouter, monkeypatch
):
    monkeypatch.setenv("TRANSLATOR_MODEL_CATALOG_TTL", "0")
    route = mock_core.get(_MODELS).respond(json=catalog())
    await get_deployment_profile(_CORE, _KEY, "gpt-5.5")
    await get_deployment_profile(_CORE, _KEY, "gpt-5.5")
    assert route.call_count == 2


async def test_the_catalog_cache_is_bounded(
    mock_core: respx.MockRouter, monkeypatch
):
    monkeypatch.setenv("TRANSLATOR_MODEL_CATALOG_SIZE", "2")
    route = mock_core.get(_MODELS).respond(json=catalog())

    for credential in ("one", "two", "three"):
        await get_deployment_profile(_CORE, ("api-key", credential), "gpt-5.5")
    # The least recently used entry ("one") was evicted, so it refetches.
    await get_deployment_profile(_CORE, ("api-key", "one"), "gpt-5.5")
    assert route.call_count == 4
