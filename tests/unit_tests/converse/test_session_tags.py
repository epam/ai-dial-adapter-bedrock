import logging

import pytest
from aidial_client import UserInfo

from aidial_adapter_bedrock.llm.converse import session_tags
from aidial_adapter_bedrock.upstream_config import (
    ApiKeyUpstreamConfig,
    AWSAssumeRoleCredentials,
    AWSClientCredentials,
    CloudUpstreamConfig,
)


def _paths(paths: str | None) -> list[str] | None:
    if paths is None:
        return None
    return [path.strip() for path in paths.split(",")]


def _assume_role_upstream_config() -> CloudUpstreamConfig:
    return CloudUpstreamConfig(
        region="us-east-1",
        claude_client="legacy",
        credentials=AWSAssumeRoleCredentials(aws_assume_role_arn="arn"),
    )


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
        ("userId", {"userId": "sub"}),
        ("roles", {"roles": '["user"]'}),
        (
            "userClaims.id,userClaims.iat",
            {"userClaims.id": "15", "userClaims.iat": "1713355825"},
        ),
        (
            "userClaims.access.0,userClaims.access.1",
            {
                "userClaims.access.0": "read",
                "userClaims.access.1": "write",
            },
        ),
        ("userClaims.map", {"userClaims.map": '{"a": ["b"]}'}),
        (
            "userId,userClaims.id,userClaims.access.0",
            {
                "userId": "sub",
                "userClaims.id": "15",
                "userClaims.access.0": "read",
            },
        ),
        (
            "userId,,userClaims.id,",
            {"userId": "sub", "userClaims.id": "15"},
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
            {"userId": "sub", "userClaims.id": "15"},
        ),
    ],
)
def test_resolve_paths_resolves_configured_paths(
    jwt_auth: dict, config: str | None, expected: dict[str, str]
):
    assert session_tags.resolve_paths(jwt_auth, _paths(config)) == expected


def test_resolve_paths_values_are_strings(jwt_auth: dict):
    out = session_tags.resolve_paths(
        jwt_auth, _paths("userId,userClaims.id,userClaims.access.0")
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
            {"s": "hi", "lst": "[1, 2]", "obj": '{"k": "v"}'},
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
            {"a.b.c.d.0": "leaf"},
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
    assert session_tags.resolve_paths(data, _paths(config)) == expected


def test_resolve_paths_logs_unresolved_path_error(caplog, jwt_auth: dict):
    caplog.set_level(logging.WARNING, logger="bedrock")

    assert session_tags.resolve_paths(jwt_auth, _paths("userClaims.nope")) == {}

    assert caplog.messages == [
        "Skipping unresolved AWS STS session tags path "
        "'userClaims.nope': KeyError: 'nope'"
    ]


@pytest.mark.parametrize(
    ("config", "expected"),
    [(None, False), ("", True), ("   ", True), ("roles.0", True)],
)
def test_is_enabled(monkeypatch: pytest.MonkeyPatch, config, expected: bool):
    monkeypatch.setattr(
        session_tags, "CONVERSE_API_SESSION_TAGS_FIELDS", _paths(config)
    )

    assert session_tags.is_enabled(_assume_role_upstream_config()) is expected


def test_is_enabled_requires_assume_role_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        session_tags, "CONVERSE_API_SESSION_TAGS_FIELDS", ["roles.0"]
    )

    assert (
        session_tags.is_enabled(
            CloudUpstreamConfig(region="us-east-1", claude_client="legacy")
        )
        is False
    )
    assert (
        session_tags.is_enabled(
            CloudUpstreamConfig(
                region="us-east-1",
                claude_client="legacy",
                credentials=AWSClientCredentials(
                    aws_access_key_id="id",
                    aws_secret_access_key="secret",  # noqa: S106
                ),
            )
        )
        is False
    )
    assert session_tags.is_enabled(ApiKeyUpstreamConfig(api_key="key")) is False


def test_to_session_tags_truncates_long_keys_and_values():
    long_key = "k" * 200
    long_value = "v" * 300

    assert session_tags._to_session_tags(
        {"bad[key]": '"a\'b"', long_key: long_value}
    ) == [
        {"Key": "bad[key]", "Value": '"a\'b"'},
        {"Key": "k" * 128, "Value": "v" * 256},
    ]


def test_to_session_tags_preserves_configured_paths_and_values():
    assert session_tags._to_session_tags({"a,b": "x", "c#d": "y$z"}) == [
        {"Key": "a,b", "Value": "x"},
        {"Key": "c#d", "Value": "y$z"},
    ]


def test_to_session_tags_caps_at_50_entries():
    flat = {f"k{i}": "v" for i in range(52)}

    assert session_tags._to_session_tags(flat) == [
        {"Key": f"k{i}", "Value": "v"} for i in range(50)
    ]


def test_to_session_tags_keeps_first_truncated_key_collision():
    first_key = f"{'a' * 128}x"
    second_key = f"{'a' * 128}y"

    assert session_tags._to_session_tags(
        {first_key: "first", second_key: "second"}
    ) == [{"Key": "a" * 128, "Value": "first"}]


def test_to_session_tags_drops_empty_keys_but_keeps_empty_values():
    assert session_tags._to_session_tags({"": "value", "empty": ""}) == [
        {"Key": "empty", "Value": ""}
    ]


def test_to_session_tags_logs_truncated_keys_and_values(caplog):
    caplog.set_level(logging.WARNING, logger="bedrock")
    long_key = "k" * 200
    long_value = "v" * 300

    assert session_tags._to_session_tags({long_key: long_value}) == [
        {"Key": "k" * 128, "Value": "v" * 256}
    ]

    assert caplog.messages == [
        f"Sanitized AWS STS session tags key(s): {long_key}",
        f"Sanitized AWS STS session tags value(s) for path(s): {long_key}",
    ]


def test_from_user_info_resolves_paths_and_sanitizes(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    monkeypatch.setattr(
        session_tags,
        "CONVERSE_API_SESSION_TAGS_FIELDS",
        _paths("roles.0,project,userClaims.id,userClaims.email,userClaims.map"),
    )

    assert session_tags.from_user_info(user_info) == [
        {"Key": "roles.0", "Value": "admin"},
        {"Key": "project", "Value": "null"},
        {"Key": "userClaims.id", "Value": "15"},
        {"Key": "userClaims.email", "Value": "user@example.com"},
        {"Key": "userClaims.map", "Value": '{"a": ["b"]}'},
    ]


class _FakeUserApi:
    def __init__(self, user_info: UserInfo | None, error: Exception | None):
        self._user_info = user_info
        self._error = error

    async def info(self) -> UserInfo:
        if self._error is not None:
            raise self._error
        assert self._user_info is not None
        return self._user_info


class _FakeDialClient:
    def __init__(
        self,
        user_info: UserInfo | None = None,
        error: Exception | None = None,
    ):
        self.user = _FakeUserApi(user_info, error)


async def test_resolve_session_tags_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(session_tags, "CONVERSE_API_SESSION_TAGS_FIELDS", None)

    assert (
        await session_tags.resolve_session_tags(
            "key", _assume_role_upstream_config()
        )
        is None
    )


async def test_resolve_session_tags_no_dial_client(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        session_tags, "CONVERSE_API_SESSION_TAGS_FIELDS", _paths("roles.0")
    )
    monkeypatch.setattr(
        session_tags, "create_dial_client", lambda api_key: None
    )

    assert (
        await session_tags.resolve_session_tags(
            "key", _assume_role_upstream_config()
        )
        is None
    )


async def test_resolve_session_tags_returns_tags(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    monkeypatch.setattr(
        session_tags,
        "CONVERSE_API_SESSION_TAGS_FIELDS",
        _paths("roles.0,userClaims.email"),
    )
    monkeypatch.setattr(
        session_tags,
        "create_dial_client",
        lambda api_key: _FakeDialClient(user_info=user_info),
    )

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config()
    ) == [
        {"Key": "roles.0", "Value": "admin"},
        {"Key": "userClaims.email", "Value": "user@example.com"},
    ]


async def test_resolve_session_tags_swallows_dial_errors(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    caplog.set_level(logging.WARNING, logger="bedrock")
    monkeypatch.setattr(
        session_tags, "CONVERSE_API_SESSION_TAGS_FIELDS", _paths("roles.0")
    )
    monkeypatch.setattr(
        session_tags,
        "create_dial_client",
        lambda api_key: _FakeDialClient(error=RuntimeError("boom")),
    )

    assert (
        await session_tags.resolve_session_tags(
            "key", _assume_role_upstream_config()
        )
        is None
    )
    assert any(
        "failed to fetch DIAL user info" in message
        for message in caplog.messages
    )
