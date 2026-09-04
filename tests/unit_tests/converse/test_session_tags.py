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
    [
        (None, False),
        ("", False),
        ("Bedrock.modelId", True),
        ("UserInfo.project", True),
        # Setting the variable enables the feature, even if every configured
        # tag turns out to be unusable.
        ("Nope.field", True),
    ],
)
def test_is_enabled_follows_the_tags_var(
    monkeypatch: pytest.MonkeyPatch, config: str | None, expected: bool
):
    monkeypatch.setattr(session_tags, "AWS_SESSION_TAGS", _paths(config))

    assert session_tags.is_enabled(_assume_role_upstream_config()) is expected


def test_is_enabled_requires_assume_role_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(session_tags, "AWS_SESSION_TAGS", ["Bedrock.modelId"])

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

    assert session_tags._to_session_tags({long_key: long_value}) == [
        {"Key": "k" * 128, "Value": "v" * 256}
    ]


@pytest.mark.parametrize(
    ("flat", "expected"),
    [
        # The AssumeRole failure this sanitization was added for: a claim
        # holding a single email in a list.
        (
            {"UserInfo.userClaims.email": '["test_user@example.com"]'},
            [
                {
                    "Key": "UserInfo.userClaims.email",
                    "Value": "__test_user@example.com__",
                }
            ],
        ),
        # A comma isn't allowed either, so no JSON value passes as-is.
        (
            {"UserInfo.userClaims.access": '["read", "write"]'},
            [
                {
                    "Key": "UserInfo.userClaims.access",
                    "Value": "__read__ _write__",
                }
            ],
        ),
        ({"a,b": "x"}, [{"Key": "a_b", "Value": "x"}]),
        ({"c#d": "y$z"}, [{"Key": "c_d", "Value": "y_z"}]),
        ({"k": '"a\'b"'}, [{"Key": "k", "Value": "_a_b_"}]),
        # The allowed punctuation survives.
        (
            {"UserInfo.project": "a_b.c:d/e=f+g-h@i"},
            [{"Key": "UserInfo.project", "Value": "a_b.c:d/e=f+g-h@i"}],
        ),
        # Letters, numbers and separators of any script survive.
        (
            {"UserInfo.project": "Ünïcode Проект 42"},
            [{"Key": "UserInfo.project", "Value": "Ünïcode Проект 42"}],
        ),
    ],
)
def test_to_session_tags_sanitizes_disallowed_chars(
    flat: dict[str, str], expected: list[dict[str, str]]
):
    assert session_tags._to_session_tags(flat) == expected


def test_to_session_tags_sanitization_preserves_length():
    value = '{"a": ["b"], "c": 1}'

    tags = session_tags._to_session_tags({"k": value})

    assert len(tags[0]["Value"]) == len(value)


def test_to_session_tags_caps_at_50_entries():
    flat = {f"k{i}": "v" for i in range(52)}

    assert session_tags._to_session_tags(flat) == [
        {"Key": f"k{i}", "Value": "v"} for i in range(50)
    ]


def test_to_session_tags_postfixes_truncated_key_collisions():
    keys = {f"{'a' * 128}{suffix}": suffix for suffix in ("x", "y", "z")}

    tags = session_tags._to_session_tags(keys)

    assert tags == [
        {"Key": "a" * 128, "Value": "x"},
        {"Key": "a" * 126 + "_1", "Value": "y"},
        {"Key": "a" * 126 + "_2", "Value": "z"},
    ]
    assert all(len(tag["Key"]) == 128 for tag in tags)


def test_to_session_tags_postfixes_sanitized_key_collisions():
    tags = session_tags._to_session_tags(
        {"a#b": "first", "a$b": "second", "a%b": "third"}
    )

    assert tags == [
        {"Key": "a_b", "Value": "first"},
        {"Key": "a_b_1", "Value": "second"},
        {"Key": "a_b_2", "Value": "third"},
    ]


def test_to_session_tags_logs_postfixed_key_collisions(caplog):
    caplog.set_level(logging.WARNING, logger="bedrock")

    session_tags._to_session_tags({"a#b": "first", "a$b": "second"})

    assert any(
        "collides with an earlier entry: a$b" in message
        for message in caplog.messages
    )


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


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, []),
        ("", []),
        ("Bedrock.modelId", [("Bedrock.modelId", "Bedrock", "modelId")]),
        (
            "UserInfo.userClaims.email",
            [("UserInfo.userClaims.email", "UserInfo", "userClaims.email")],
        ),
        # The order of the variable is kept.
        (
            "UserInfo.project,Bedrock.modelId",
            [
                ("UserInfo.project", "UserInfo", "project"),
                ("Bedrock.modelId", "Bedrock", "modelId"),
            ],
        ),
        # An unprefixed entry names no source.
        ("project", []),
        ("", []),
        # Unknown source.
        ("Nope.project", []),
        # The Bedrock source only provides modelId.
        ("Bedrock.region", []),
        (
            "Bedrock.region,UserInfo.project",
            [("UserInfo.project", "UserInfo", "project")],
        ),
    ],
)
def test_parse_tags(config: str | None, expected: list[tuple[str, str, str]]):
    assert session_tags.parse_tags(_paths(config)) == expected


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("project", "it names no source"),
        ("Nope.project", "unknown source 'Nope'"),
        ("Bedrock.region", "only provides 'modelId'"),
    ],
)
def test_parse_tags_logs_skipped_tags(caplog, config: str, message: str):
    caplog.set_level(logging.WARNING, logger="bedrock")

    assert session_tags.parse_tags(_paths(config)) == []

    assert any(message in logged for logged in caplog.messages)


def test_build_tags_resolves_configured_tags(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths(
            "Bedrock.modelId,UserInfo.roles.0,UserInfo.project,"
            "UserInfo.userClaims.id,UserInfo.userClaims.email,"
            "UserInfo.userClaims.map"
        ),
    )

    assert session_tags.build_tags("my-claude", user_info) == [
        {"Key": "Bedrock.modelId", "Value": "my-claude"},
        {"Key": "UserInfo.roles.0", "Value": "admin"},
        {"Key": "UserInfo.project", "Value": "null"},
        {"Key": "UserInfo.userClaims.id", "Value": "15"},
        {"Key": "UserInfo.userClaims.email", "Value": "user@example.com"},
        # The JSON punctuation isn't allowed by AWS.
        {"Key": "UserInfo.userClaims.map", "Value": "__a_: __b___"},
    ]


def test_build_tags_keeps_the_configured_order(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("UserInfo.roles.1,Bedrock.modelId,UserInfo.roles.0"),
    )

    assert session_tags.build_tags("my-claude", user_info) == [
        {"Key": "UserInfo.roles.1", "Value": "writer"},
        {"Key": "Bedrock.modelId", "Value": "my-claude"},
        {"Key": "UserInfo.roles.0", "Value": "admin"},
    ]


def test_build_tags_passes_only_the_configured_tags(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    """A tag that isn't configured is never passed, model id included."""

    monkeypatch.setattr(
        session_tags, "AWS_SESSION_TAGS", _paths("UserInfo.roles.0")
    )

    assert session_tags.build_tags("my-claude", user_info) == [
        {"Key": "UserInfo.roles.0", "Value": "admin"}
    ]


def test_build_tags_without_user_info(monkeypatch: pytest.MonkeyPatch):
    """An unavailable UserInfo mustn't sink the Bedrock tags."""

    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("Bedrock.modelId,UserInfo.roles.0"),
    )

    assert session_tags.build_tags("my-claude", None) == [
        {"Key": "Bedrock.modelId", "Value": "my-claude"}
    ]


def test_build_tags_without_a_model_id(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    """An unknown model mustn't sink the UserInfo tags."""

    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("Bedrock.modelId,UserInfo.roles.0"),
    )

    assert session_tags.build_tags(None, user_info) == [
        {"Key": "UserInfo.roles.0", "Value": "admin"}
    ]


def test_build_tags_without_any_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("Bedrock.modelId,UserInfo.roles.0"),
    )

    assert session_tags.build_tags(None, None) == []


async def test_resolve_session_tags_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """A request with no DIAL API key still gets the other tag sources."""

    caplog.set_level(logging.WARNING, logger="bedrock")
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("Bedrock.modelId,UserInfo.roles.0"),
    )

    assert await session_tags.resolve_session_tags(
        None, _assume_role_upstream_config(), "my-claude"
    ) == [{"Key": "Bedrock.modelId", "Value": "my-claude"}]
    assert any(
        "carries no DIAL API key" in message for message in caplog.messages
    )


async def test_resolve_session_tags_without_any_source(
    monkeypatch: pytest.MonkeyPatch,
):
    """No tags at all is reported as None, so AssumeRole omits Tags."""

    monkeypatch.setattr(
        session_tags, "AWS_SESSION_TAGS", _paths("Bedrock.modelId")
    )

    assert (
        await session_tags.resolve_session_tags(
            None, _assume_role_upstream_config(), None
        )
        is None
    )


def test_build_tags_caps_at_50_entries(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        ["Bedrock.modelId"] + [f"UserInfo.roles.{i}" for i in range(60)],
    )

    tags = session_tags.build_tags(
        "my-claude", UserInfo(roles=[f"r{i}" for i in range(60)])
    )

    assert tags == [{"Key": "Bedrock.modelId", "Value": "my-claude"}] + [
        {"Key": f"UserInfo.roles.{i}", "Value": f"r{i}"} for i in range(49)
    ]


def test_build_tags_truncates_long_model_ids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        session_tags, "AWS_SESSION_TAGS", _paths("Bedrock.modelId")
    )

    assert session_tags.build_tags("d" * 300, None) == [
        {"Key": "Bedrock.modelId", "Value": "d" * 256}
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
    monkeypatch.setattr(session_tags, "AWS_SESSION_TAGS", None)

    assert (
        await session_tags.resolve_session_tags(
            "key", _assume_role_upstream_config(), "my-claude"
        )
        is None
    )


async def test_resolve_session_tags_without_user_info_tags(
    monkeypatch: pytest.MonkeyPatch,
):
    """The UserInfo request isn't made at all when no UserInfo tag is asked
    for."""

    monkeypatch.setattr(
        session_tags, "AWS_SESSION_TAGS", _paths("Bedrock.modelId")
    )

    def _unexpected_client(api_key: str):
        raise AssertionError("the DIAL client must not be created")

    monkeypatch.setattr(session_tags, "create_dial_client", _unexpected_client)

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [{"Key": "Bedrock.modelId", "Value": "my-claude"}]


async def test_resolve_session_tags_no_dial_client(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """DIAL_URL isn't set: the other tag sources are still passed."""

    caplog.set_level(logging.WARNING, logger="bedrock")
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("Bedrock.modelId,UserInfo.roles.0"),
    )
    monkeypatch.setattr(
        session_tags, "create_dial_client", lambda api_key: None
    )

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [{"Key": "Bedrock.modelId", "Value": "my-claude"}]
    assert any(
        "DIAL_URL env variable is not set" in message
        for message in caplog.messages
    )


async def test_resolve_session_tags_returns_tags(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("Bedrock.modelId,UserInfo.roles.0,UserInfo.userClaims.email"),
    )
    monkeypatch.setattr(
        session_tags,
        "create_dial_client",
        lambda api_key: _FakeDialClient(user_info=user_info),
    )

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [
        {"Key": "Bedrock.modelId", "Value": "my-claude"},
        {"Key": "UserInfo.roles.0", "Value": "admin"},
        {"Key": "UserInfo.userClaims.email", "Value": "user@example.com"},
    ]


async def test_resolve_session_tags_swallows_dial_errors(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """A failed UserInfo request doesn't drop the other tag sources."""

    caplog.set_level(logging.WARNING, logger="bedrock")
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        _paths("Bedrock.modelId,UserInfo.roles.0"),
    )
    monkeypatch.setattr(
        session_tags,
        "create_dial_client",
        lambda api_key: _FakeDialClient(error=RuntimeError("boom")),
    )

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [{"Key": "Bedrock.modelId", "Value": "my-claude"}]
    assert any(
        "failed to fetch DIAL user info" in message
        for message in caplog.messages
    )
