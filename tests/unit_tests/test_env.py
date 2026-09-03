import pytest

from aidial_adapter_bedrock.utils.env import get_env_bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("  TRUE  ", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("yes", False),
    ],
)
def test_get_env_bool(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
):
    monkeypatch.setenv("SOME_FLAG", value)

    assert get_env_bool("SOME_FLAG") is expected


@pytest.mark.parametrize("default", [False, True])
def test_get_env_bool_default(monkeypatch: pytest.MonkeyPatch, default: bool):
    monkeypatch.delenv("SOME_FLAG", raising=False)

    assert get_env_bool("SOME_FLAG", default) is default
