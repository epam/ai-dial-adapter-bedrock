import pytest
from aidial_client import UserInfo

from aidial_adapter_bedrock.llm.converse import request_metadata


@pytest.fixture
def user_info() -> UserInfo:
    return UserInfo(
        roles=["admin"],
        project="test-project",
        userClaims={"email": "user@example.com"},
    )


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (None, {}),
        ("", {}),
        (
            "*",
            {
                "roles": "",
                "project": "",
                "userClaims": "",
            },
        ),
        ("roles", {"roles": ""}),
        (
            "roles,project,userClaims",
            {"roles": "", "project": "", "userClaims": ""},
        ),
        ("roles, userClaims", {"roles": "", "userClaims": ""}),
        (" roles , project ", {"roles": "", "project": ""}),
    ],
)
def test_request_metadata_fields(
    monkeypatch: pytest.MonkeyPatch,
    fields: str | None,
    expected: dict[str, str],
):
    monkeypatch.setattr(
        request_metadata, "CONVERSE_API_REQUEST_METADATA_FIELDS", fields
    )

    assert request_metadata.request_metadata_fields() == expected


@pytest.mark.parametrize(
    ("fields", "expected_error"),
    [
        ("unknownField", "Unknown UserInfo fields: unknownField"),
        (
            "roles,project,unknownField",
            "Unknown UserInfo fields: unknownField",
        ),
        (
            "roles, unknownField, anotherUnknownField",
            "Unknown UserInfo fields: anotherUnknownField, unknownField",
        ),
    ],
)
def test_request_metadata_fields_invalid(
    monkeypatch: pytest.MonkeyPatch,
    fields: str,
    expected_error: str,
):
    monkeypatch.setattr(
        request_metadata, "CONVERSE_API_REQUEST_METADATA_FIELDS", fields
    )

    with pytest.raises(ValueError, match=expected_error):
        request_metadata.request_metadata_fields()


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (None, {}),
        ("", {}),
        (
            "*",
            {
                "roles": ["admin"],
                "project": "test-project",
                "userClaims": {"email": "user@example.com"},
            },
        ),
        ("roles, project", {"roles": ["admin"], "project": "test-project"}),
    ],
)
def test_filter_user_info(
    monkeypatch: pytest.MonkeyPatch,
    user_info: UserInfo,
    fields: str | None,
    expected: dict,
):
    monkeypatch.setattr(
        request_metadata,
        "CONVERSE_API_REQUEST_METADATA_FIELDS",
        fields,
    )

    assert request_metadata.from_user_info(user_info) == expected


def test_filter_user_info_raises_for_invalid_fields(
    monkeypatch: pytest.MonkeyPatch, user_info: UserInfo
):
    monkeypatch.setattr(
        request_metadata,
        "CONVERSE_API_REQUEST_METADATA_FIELDS",
        "roles, unknownField",
    )

    with pytest.raises(
        ValueError, match="Unknown UserInfo fields: unknownField"
    ):
        request_metadata.from_user_info(user_info)
