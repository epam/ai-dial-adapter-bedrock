import logging

import pytest
from aidial_client import UserInfo

from aidial_adapter_bedrock.llm.converse import request_metadata


@pytest.fixture
def user_info() -> UserInfo:
    return UserInfo(
        roles=["admin", "writer"],
        project=None,
        userClaims={
            "email": "user@example.com",
            "id": 15,
            "access": ["read", "write"],
            "map": {"a": ["b"]},
        },
    )


@pytest.fixture
def jwt_auth() -> dict:
    return {
        "roles": ["user"],
        "userId": "sub",
        "userClaims": {
            "roles": ["role"],
            "email": ["test@email.com"],
            "id": 15,
            "access": ["read", "write"],
            "map": {"a": ["b"]},
            "sub": ["sub"],
            "iat": 1713355825,
        },
    }


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, {}),
        ("", {}),
        ("   ", {}),
        ("*", {}),
        ("userId", {"userId": '"sub"'}),
        ("roles", {"roles": '["user"]'}),
        (
            "userClaims.id,userClaims.iat",
            {"userClaims.id": "15", "userClaims.iat": "1713355825"},
        ),
        (
            "userClaims.access.0,userClaims.access.1",
            {
                "userClaims.access.0": '"read"',
                "userClaims.access.1": '"write"',
            },
        ),
        ("userClaims.map", {"userClaims.map": '{"a": ["b"]}'}),
        (
            "userId,userClaims.id,userClaims.access.0",
            {
                "userId": '"sub"',
                "userClaims.id": "15",
                "userClaims.access.0": '"read"',
            },
        ),
        (
            "userId,,userClaims.id,",
            {"userId": '"sub"', "userClaims.id": "15"},
        ),
        (
            "  userClaims.id ,  userClaims.iat ",
            {"userClaims.id": "15", "userClaims.iat": "1713355825"},
        ),
        ("userClaims.nope", {}),
        ("userClaims.access.99", {}),
        ("userId.0", {}),
        ("roles.x", {}),
        (
            "userId,does.not.exist,userClaims.id",
            {"userId": '"sub"', "userClaims.id": "15"},
        ),
    ],
)
def test_resolve_paths_resolves_configured_paths(
    jwt_auth: dict, config: str | None, expected: dict[str, str]
):
    assert request_metadata.resolve_paths(jwt_auth, config) == expected


def test_resolve_paths_values_are_strings(jwt_auth: dict):
    out = request_metadata.resolve_paths(
        jwt_auth, "userId,userClaims.id,userClaims.access.0"
    )

    assert out
    assert all(isinstance(value, str) for value in out.values())


@pytest.mark.parametrize(
    ("data", "config", "expected"),
    [
        (
            {"n": 15, "flag": True, "off": False, "missing": None},
            "n,flag,off,missing",
            {
                "n": "15",
                "flag": "true",
                "off": "false",
                "missing": "null",
            },
        ),
        (
            {"s": "hi", "lst": [1, 2], "obj": {"k": "v"}},
            "s,lst,obj",
            {"s": '"hi"', "lst": "[1, 2]", "obj": '{"k": "v"}'},
        ),
        (
            {"field1": [{"field2": [10, 20, 30]}]},
            "field1.0.field2.2",
            {"field1.0.field2.2": "30"},
        ),
        ({}, "userId,anything", {}),
        (
            {"a": {"b": {"c": {"d": ["leaf"]}}}},
            "a.b.c.d.0",
            {"a.b.c.d.0": '"leaf"'},
        ),
        (
            {"a": {"b": {"c": {"d": ["leaf"]}}}},
            "a.b.c",
            {"a.b.c": '{"d": ["leaf"]}'},
        ),
    ],
)
def test_resolve_paths_serializes_json_values(
    data: dict, config: str, expected: dict[str, str]
):
    assert request_metadata.resolve_paths(data, config) == expected


def test_resolve_paths_logs_unresolved_path_error(caplog, jwt_auth: dict):
    caplog.set_level(logging.WARNING, logger="bedrock")

    assert request_metadata.resolve_paths(jwt_auth, "userClaims.nope") == {}

    assert "userClaims.nope" in caplog.text
    assert "KeyError" in caplog.text


@pytest.mark.parametrize(
    ("config", "expected"),
    [(None, False), ("", False), ("   ", False), ("roles.0", True)],
)
def test_is_enabled(monkeypatch: pytest.MonkeyPatch, config, expected: bool):
    monkeypatch.setattr(
        request_metadata, "CONVERSE_API_REQUEST_METADATA_FIELDS", config
    )

    assert request_metadata.is_enabled() is expected


def test_to_bedrock_metadata_sanitizes_charset_and_truncates():
    long_key = "k" * 300
    long_value = "v" * 300

    assert request_metadata._to_bedrock_metadata(
        {"bad[key]": '"a\'b"', long_key: long_value}
    ) == {
        "badkey": "ab",
        "k" * 256: "v" * 256,
    }


def test_to_bedrock_metadata_caps_at_16_entries():
    flat = {f"k{i}": "v" for i in range(18)}

    assert request_metadata._to_bedrock_metadata(flat) == {
        f"k{i}": "v" for i in range(16)
    }


def test_to_bedrock_metadata_keeps_first_sanitized_key_collision():
    assert request_metadata._to_bedrock_metadata(
        {"a[": "first", "a]": "second"}
    ) == {"a": "first"}


def test_to_bedrock_metadata_drops_empty_keys_but_keeps_empty_values():
    assert request_metadata._to_bedrock_metadata(
        {"[]": "value", "empty": '""'}
    ) == {"empty": ""}


def test_to_bedrock_metadata_logs_paths_without_values(caplog):
    caplog.set_level(logging.WARNING, logger="bedrock")

    assert request_metadata._to_bedrock_metadata({"bad[key]": '"secret"'}) == {
        "badkey": "secret"
    }

    assert "bad[key]" in caplog.text
    assert "secret" not in caplog.text


def test_from_user_info_resolves_paths_and_sanitizes(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    monkeypatch.setattr(
        request_metadata,
        "CONVERSE_API_REQUEST_METADATA_FIELDS",
        "roles.0,project,userClaims.id,userClaims.email,userClaims.map",
    )

    assert request_metadata.from_user_info(user_info) == {
        "roles.0": "admin",
        "project": "null",
        "userClaims.id": "15",
        "userClaims.email": "user@example.com",
        "userClaims.map": "a: b",
    }
