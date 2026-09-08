import logging

import pytest
from aidial_client import UserInfo

from aidial_adapter_bedrock.upstream_config import (
    ApiKeyUpstreamConfig,
    AWSAssumeRoleCredentials,
    AWSClientCredentials,
    CloudUpstreamConfig,
)
from aidial_adapter_bedrock.utils import session_tags
from aidial_adapter_bedrock.utils.session_tags import Tags


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
        ({}, False),
        ({"application": "Bedrock.modelId"}, True),
        ({"project": "UserInfo.project"}, True),
        # Setting the variable enables the feature, even if every configured
        # tag turns out to be unusable.
        ({"nope": "Nope.field"}, True),
    ],
)
def test_is_enabled_follows_the_tags_var(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, str], expected: bool
):
    monkeypatch.setattr(session_tags, "AWS_SESSION_TAGS", config)

    assert session_tags.is_enabled(_assume_role_upstream_config()) is expected


def test_is_enabled_requires_assume_role_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        session_tags, "AWS_SESSION_TAGS", {"application": "Bedrock.modelId"}
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


def test_to_session_tags_fits_the_alias_not_the_path():
    """The alias is the AWS tag key, so it's the one AWS constrains."""

    long_alias = "a" * 200
    long_value = "v" * 300

    assert session_tags._to_session_tags(
        {"UserInfo.project": long_value}, {"UserInfo.project": long_alias}
    ) == [
        {
            "Key": "UserInfo.project",
            "KeyAlias": "a" * 128,
            "Value": "v" * 256,
        }
    ]


def test_to_session_tags_falls_back_to_the_path_without_an_alias():
    assert session_tags._to_session_tags({"UserInfo.project": "epam"}, {}) == [
        {
            "Key": "UserInfo.project",
            "KeyAlias": "UserInfo.project",
            "Value": "epam",
        }
    ]


@pytest.mark.parametrize(
    ("flat", "aliases", "expected"),
    [
        # The AssumeRole failure this sanitization was added for: a claim
        # holding a single email in a list.
        (
            {"UserInfo.userClaims.email": '["test_user@example.com"]'},
            {"UserInfo.userClaims.email": "employee"},
            [
                {
                    "Key": "UserInfo.userClaims.email",
                    "KeyAlias": "employee",
                    "Value": "__test_user@example.com__",
                }
            ],
        ),
        # A comma isn't allowed either, so no JSON value passes as-is.
        (
            {"UserInfo.userClaims.access": '["read", "write"]'},
            {"UserInfo.userClaims.access": "access"},
            [
                {
                    "Key": "UserInfo.userClaims.access",
                    "KeyAlias": "access",
                    "Value": "__read__ _write__",
                }
            ],
        ),
        # A disallowed character in the alias is replaced.
        (
            {"p": "x"},
            {"p": "a,b"},
            [{"Key": "p", "KeyAlias": "a_b", "Value": "x"}],
        ),
        (
            {"p": "y$z"},
            {"p": "c#d"},
            [{"Key": "p", "KeyAlias": "c_d", "Value": "y_z"}],
        ),
        (
            {"p": '"a\'b"'},
            {"p": "k"},
            [{"Key": "p", "KeyAlias": "k", "Value": "_a_b_"}],
        ),
        # The allowed punctuation survives.
        (
            {"UserInfo.project": "a_b.c:d/e=f+g-h@i"},
            {"UserInfo.project": "a_b.c:d/e=f+g-h@i"},
            [
                {
                    "Key": "UserInfo.project",
                    "KeyAlias": "a_b.c:d/e=f+g-h@i",
                    "Value": "a_b.c:d/e=f+g-h@i",
                }
            ],
        ),
        # Letters, numbers and separators of any script survive.
        (
            {"UserInfo.project": "Ünïcode Проект 42"},
            {"UserInfo.project": "Проект"},
            [
                {
                    "Key": "UserInfo.project",
                    "KeyAlias": "Проект",
                    "Value": "Ünïcode Проект 42",
                }
            ],
        ),
    ],
)
def test_to_session_tags_sanitizes_disallowed_chars(
    flat: dict[str, str],
    aliases: dict[str, str],
    expected: list[dict[str, str]],
):
    assert session_tags._to_session_tags(flat, aliases) == expected


def test_to_session_tags_sanitization_preserves_length():
    value = '{"a": ["b"], "c": 1}'

    tags = session_tags._to_session_tags({"k": value}, {"k": "alias"})

    assert len(tags[0]["Value"]) == len(value)


def test_to_session_tags_caps_at_50_entries():
    flat = {f"p{i}": "v" for i in range(52)}
    aliases = {f"p{i}": f"a{i}" for i in range(52)}

    assert session_tags._to_session_tags(flat, aliases) == [
        {"Key": f"p{i}", "KeyAlias": f"a{i}", "Value": "v"} for i in range(50)
    ]


def test_to_session_tags_logs_the_capped_entries_by_alias(caplog):
    caplog.set_level(logging.WARNING, logger="bedrock")

    session_tags._to_session_tags(
        {f"p{i}": "v" for i in range(52)},
        {f"p{i}": f"a{i}" for i in range(52)},
    )

    assert any(
        "omitted 2 configured tag(s): a50, a51" in message
        for message in caplog.messages
    )


def test_to_session_tags_postfixes_truncated_alias_collisions():
    flat = {suffix: suffix for suffix in ("x", "y", "z")}
    aliases = {suffix: f"{'a' * 128}{suffix}" for suffix in ("x", "y", "z")}

    tags = session_tags._to_session_tags(flat, aliases)

    assert tags == [
        {"Key": "x", "KeyAlias": "a" * 128, "Value": "x"},
        {"Key": "y", "KeyAlias": "a" * 126 + "_1", "Value": "y"},
        {"Key": "z", "KeyAlias": "a" * 126 + "_2", "Value": "z"},
    ]
    assert all(len(tag["KeyAlias"]) == 128 for tag in tags)


def test_to_session_tags_postfixes_sanitized_alias_collisions():
    tags = session_tags._to_session_tags(
        {"p1": "first", "p2": "second", "p3": "third"},
        {"p1": "a#b", "p2": "a$b", "p3": "a%b"},
    )

    assert tags == [
        {"Key": "p1", "KeyAlias": "a_b", "Value": "first"},
        {"Key": "p2", "KeyAlias": "a_b_1", "Value": "second"},
        {"Key": "p3", "KeyAlias": "a_b_2", "Value": "third"},
    ]


def test_to_session_tags_logs_postfixed_alias_collisions(caplog):
    caplog.set_level(logging.WARNING, logger="bedrock")

    session_tags._to_session_tags(
        {"p1": "first", "p2": "second"}, {"p1": "a#b", "p2": "a$b"}
    )

    # The alias is reported, since that's what the check ran on.
    assert any(
        "collides with an earlier entry: a$b" in message
        for message in caplog.messages
    )


def test_to_session_tags_drops_empty_aliases_but_keeps_empty_values():
    assert session_tags._to_session_tags(
        {"p1": "value", "p2": ""}, {"p1": "", "p2": "empty"}
    ) == [{"Key": "p2", "KeyAlias": "empty", "Value": ""}]


def test_to_session_tags_logs_the_field_of_an_empty_alias(caplog):
    """An empty alias can't name itself, so the field identifies the entry."""

    caplog.set_level(logging.WARNING, logger="bedrock")

    session_tags._to_session_tags(
        {"UserInfo.project": "epam"}, {"UserInfo.project": ""}
    )

    assert any(
        "empty key, configured for field(s): UserInfo.project" in message
        for message in caplog.messages
    )


def test_to_session_tags_logs_truncated_aliases_and_values(caplog):
    caplog.set_level(logging.WARNING, logger="bedrock")
    long_alias = "a" * 200
    long_value = "v" * 300

    assert session_tags._to_session_tags(
        {"UserInfo.project": long_value}, {"UserInfo.project": long_alias}
    ) == [
        {
            "Key": "UserInfo.project",
            "KeyAlias": "a" * 128,
            "Value": "v" * 256,
        }
    ]

    # The alias is reported, since that's what the checks ran on.
    assert caplog.messages == [
        f"Sanitized AWS STS session tags key(s): {long_alias}",
        f"Sanitized AWS STS session tags value(s): {long_alias}",
    ]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, Tags(bedrock_model_id=False, user_info_paths=[], aliases={})),
        (
            {"application": "Bedrock.modelId"},
            Tags(
                bedrock_model_id=True,
                user_info_paths=[],
                aliases={"Bedrock.modelId": "application"},
            ),
        ),
        (
            {"employee": "UserInfo.userClaims.email"},
            Tags(
                bedrock_model_id=False,
                user_info_paths=["userClaims.email"],
                # The alias is keyed by the prefixed path, so that
                # `to_session_tags` finds it again.
                aliases={"UserInfo.userClaims.email": "employee"},
            ),
        ),
        (
            {"project": "UserInfo.project", "application": "Bedrock.modelId"},
            Tags(
                bedrock_model_id=True,
                user_info_paths=["project"],
                aliases={
                    "UserInfo.project": "project",
                    "Bedrock.modelId": "application",
                },
            ),
        ),
        # The order of the UserInfo paths is kept.
        (
            {"second": "UserInfo.roles.1", "first": "UserInfo.roles.0"},
            Tags(
                bedrock_model_id=False,
                user_info_paths=["roles.1", "roles.0"],
                aliases={
                    "UserInfo.roles.1": "second",
                    "UserInfo.roles.0": "first",
                },
            ),
        ),
        # Unknown fields: no prefix, an unknown source, a field the Bedrock
        # source doesn't provide, and a near-miss of the UserInfo prefix.
        (
            {"a": "project"},
            Tags(bedrock_model_id=False, user_info_paths=[], aliases={}),
        ),
        (
            {"a": "Nope.project"},
            Tags(bedrock_model_id=False, user_info_paths=[], aliases={}),
        ),
        (
            {"a": "Bedrock.region"},
            Tags(bedrock_model_id=False, user_info_paths=[], aliases={}),
        ),
        (
            {"a": "UserInfoProject"},
            Tags(bedrock_model_id=False, user_info_paths=[], aliases={}),
        ),
        (
            {"a": "Bedrock.region", "project": "UserInfo.project"},
            Tags(
                bedrock_model_id=False,
                user_info_paths=["project"],
                aliases={"UserInfo.project": "project"},
            ),
        ),
    ],
)
def test_tags_parse(config: dict[str, str], expected: Tags):
    assert Tags.parse(config) == expected


@pytest.mark.parametrize(
    "field", ["project", "Nope.project", "Bedrock.region", "UserInfoProject"]
)
def test_tags_parse_logs_unknown_fields(caplog, field: str):
    caplog.set_level(logging.WARNING, logger="bedrock")

    Tags.parse({"my_alias": field})

    assert any(
        f"Skipping AWS STS session tag 'my_alias': unknown field {field!r}"
        in logged
        for logged in caplog.messages
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, False),
        ({"application": "Bedrock.modelId"}, False),
        ({"project": "UserInfo.project"}, True),
        (
            {"application": "Bedrock.modelId", "project": "UserInfo.project"},
            True,
        ),
    ],
)
def test_tags_wants_user_info(config: dict[str, str], expected: bool):
    assert Tags.parse(config).wants_user_info is expected


def test_to_session_tags_keys_every_tag_by_its_alias(user_info: UserInfo):
    tags = Tags.parse(
        {
            "application": "Bedrock.modelId",
            "role": "UserInfo.roles.0",
            "project": "UserInfo.project",
            "id": "UserInfo.userClaims.id",
            "employee": "UserInfo.userClaims.email",
            "map": "UserInfo.userClaims.map",
        }
    )

    assert tags.to_session_tags("my-claude", user_info) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        },
        {"Key": "UserInfo.roles.0", "KeyAlias": "role", "Value": "admin"},
        {"Key": "UserInfo.project", "KeyAlias": "project", "Value": "null"},
        {"Key": "UserInfo.userClaims.id", "KeyAlias": "id", "Value": "15"},
        {
            "Key": "UserInfo.userClaims.email",
            "KeyAlias": "employee",
            "Value": "user@example.com",
        },
        # The JSON punctuation isn't allowed by AWS.
        {
            "Key": "UserInfo.userClaims.map",
            "KeyAlias": "map",
            "Value": "__a_: __b___",
        },
    ]


def test_to_session_tags_puts_the_model_id_first(user_info: UserInfo):
    """The model id leads, so the entry cap can never drop it."""

    tags = Tags.parse(
        {
            "second": "UserInfo.roles.1",
            "application": "Bedrock.modelId",
            "first": "UserInfo.roles.0",
        }
    )

    assert tags.to_session_tags("my-claude", user_info) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        },
        {"Key": "UserInfo.roles.1", "KeyAlias": "second", "Value": "writer"},
        {"Key": "UserInfo.roles.0", "KeyAlias": "first", "Value": "admin"},
    ]


def test_to_session_tags_passes_only_the_configured_tags(user_info: UserInfo):
    """A tag that isn't configured is never passed, model id included."""

    tags = Tags.parse({"role": "UserInfo.roles.0"})

    assert tags.to_session_tags("my-claude", user_info) == [
        {"Key": "UserInfo.roles.0", "KeyAlias": "role", "Value": "admin"}
    ]


def test_to_session_tags_without_user_info():
    """An unavailable UserInfo mustn't sink the Bedrock tags."""

    tags = Tags.parse(
        {"application": "Bedrock.modelId", "role": "UserInfo.roles.0"}
    )

    assert tags.to_session_tags("my-claude", None) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        }
    ]


def test_to_session_tags_without_a_model_id(user_info: UserInfo):
    """An unknown model mustn't sink the UserInfo tags."""

    tags = Tags.parse(
        {"application": "Bedrock.modelId", "role": "UserInfo.roles.0"}
    )

    assert tags.to_session_tags(None, user_info) == [
        {"Key": "UserInfo.roles.0", "KeyAlias": "role", "Value": "admin"}
    ]


def test_to_session_tags_without_any_source():
    tags = Tags.parse(
        {"application": "Bedrock.modelId", "role": "UserInfo.roles.0"}
    )

    assert tags.to_session_tags(None, None) == []


async def test_resolve_session_tags_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """A request with no DIAL API key still gets the other tag sources."""

    caplog.set_level(logging.WARNING, logger="bedrock")
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        {"application": "Bedrock.modelId", "role": "UserInfo.roles.0"},
    )

    assert await session_tags.resolve_session_tags(
        None, _assume_role_upstream_config(), "my-claude"
    ) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        }
    ]
    assert any(
        "carries no DIAL API key" in message for message in caplog.messages
    )


async def test_resolve_session_tags_without_any_source(
    monkeypatch: pytest.MonkeyPatch,
):
    """No tags at all is reported as None, so AssumeRole omits Tags."""

    monkeypatch.setattr(
        session_tags, "AWS_SESSION_TAGS", {"application": "Bedrock.modelId"}
    )

    assert (
        await session_tags.resolve_session_tags(
            None, _assume_role_upstream_config(), None
        )
        is None
    )


def test_to_session_tags_keeps_the_model_id_when_capped():
    tags = Tags.parse(
        {"application": "Bedrock.modelId"}
        | {f"role_{i}": f"UserInfo.roles.{i}" for i in range(60)}
    )

    assert tags.to_session_tags(
        "my-claude", UserInfo(roles=[f"r{i}" for i in range(60)])
    ) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        }
    ] + [
        {
            "Key": f"UserInfo.roles.{i}",
            "KeyAlias": f"role_{i}",
            "Value": f"r{i}",
        }
        for i in range(49)
    ]


def test_to_session_tags_truncates_long_model_ids():
    tags = Tags.parse({"application": "Bedrock.modelId"})

    assert tags.to_session_tags("d" * 300, None) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "d" * 256,
        }
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
    monkeypatch.setattr(session_tags, "AWS_SESSION_TAGS", {})

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
        session_tags, "AWS_SESSION_TAGS", {"application": "Bedrock.modelId"}
    )

    def _unexpected_client(api_key: str):
        raise AssertionError("the DIAL client must not be created")

    monkeypatch.setattr(session_tags, "create_dial_client", _unexpected_client)

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        }
    ]


async def test_resolve_session_tags_no_dial_client(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """DIAL_URL isn't set: the other tag sources are still passed."""

    caplog.set_level(logging.WARNING, logger="bedrock")
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        {"application": "Bedrock.modelId", "role": "UserInfo.roles.0"},
    )
    monkeypatch.setattr(
        session_tags, "create_dial_client", lambda api_key: None
    )

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        }
    ]
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
        {
            "application": "Bedrock.modelId",
            "role": "UserInfo.roles.0",
            "employee": "UserInfo.userClaims.email",
        },
    )
    monkeypatch.setattr(
        session_tags,
        "create_dial_client",
        lambda api_key: _FakeDialClient(user_info=user_info),
    )

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        },
        {"Key": "UserInfo.roles.0", "KeyAlias": "role", "Value": "admin"},
        {
            "Key": "UserInfo.userClaims.email",
            "KeyAlias": "employee",
            "Value": "user@example.com",
        },
    ]


async def test_resolve_session_tags_swallows_dial_errors(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """A failed UserInfo request doesn't drop the other tag sources."""

    caplog.set_level(logging.WARNING, logger="bedrock")
    monkeypatch.setattr(
        session_tags,
        "AWS_SESSION_TAGS",
        {"application": "Bedrock.modelId", "role": "UserInfo.roles.0"},
    )
    monkeypatch.setattr(
        session_tags,
        "create_dial_client",
        lambda api_key: _FakeDialClient(error=RuntimeError("boom")),
    )

    assert await session_tags.resolve_session_tags(
        "key", _assume_role_upstream_config(), "my-claude"
    ) == [
        {
            "Key": "Bedrock.modelId",
            "KeyAlias": "application",
            "Value": "my-claude",
        }
    ]
    assert any(
        "failed to fetch DIAL user info" in message
        for message in caplog.messages
    )
