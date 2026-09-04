import json
from dataclasses import dataclass
from datetime import datetime
from unittest import mock

import pytest
from pydantic import ValidationError

from aidial_adapter_bedrock.upstream_config import (
    ApiKeyUpstreamConfig,
    AWSAssumeRoleCredentials,
    ClientCredentialArgs,
    CloudUpstreamConfig,
    SessionTag,
    _get_role_session_name,
    parse_upstream_config,
)


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (None, "BedrockAccessSession"),
        ([], "BedrockAccessSession"),
        # No project tag among the passed ones.
        ([{"Key": "Bedrock.modelId", "Value": "m"}], "BedrockAccessSession"),
        # The unprefixed key isn't the project tag.
        ([{"Key": "project", "Value": "epam"}], "BedrockAccessSession"),
        # A user without a project resolves to the JSON "null".
        (
            [{"Key": "UserInfo.project", "Value": "null"}],
            "BedrockAccessSession",
        ),
        ([{"Key": "UserInfo.project", "Value": ""}], "BedrockAccessSession"),
        ([{"Key": "UserInfo.project", "Value": "epam"}], "Project_epam"),
        (
            [
                {"Key": "Bedrock.modelId", "Value": "m"},
                {"Key": "UserInfo.project", "Value": "epam"},
            ],
            "Project_epam",
        ),
        # Characters outside the RoleSessionName charset are replaced.
        (
            [{"Key": "UserInfo.project", "Value": "EPAM / DIAL (prod)"}],
            "Project_EPAM___DIAL__prod_",
        ),
        # The charset allows these as-is.
        (
            [{"Key": "UserInfo.project", "Value": "a+b=c,d.e@f-g_1"}],
            "Project_a+b=c,d.e@f-g_1",
        ),
    ],
)
def test_get_role_session_name(tags: list[SessionTag] | None, expected: str):
    assert _get_role_session_name(tags) == expected


def test_get_role_session_name_truncates_long_projects():
    name = _get_role_session_name(
        [{"Key": "UserInfo.project", "Value": "p" * 100}]
    )

    assert name == "Project_" + "p" * 56
    assert len(name) == 64


@dataclass
class OriginalRequest:
    headers: dict[str, str]


@dataclass
class FakeRequest:
    headers: dict[str, str]
    original_request: OriginalRequest


class TestAWSClientConfigFactory:
    @pytest.fixture(autouse=True)
    def _clear_default_claude_client_env(self, monkeypatch):
        # Keep default-client assertions independent from local .env values.
        monkeypatch.delenv("AWS_CLAUDE_DEFAULT_CLIENT", raising=False)

    @staticmethod
    def _get_request(
        *, extra_data: dict | None = None, api_key: str | None = None
    ):
        headers = {}
        if extra_data is not None:
            headers["x-upstream-extra-data"] = json.dumps(extra_data)
        if api_key is not None:
            headers["x-upstream-key"] = api_key
        return FakeRequest(
            headers=headers, original_request=OriginalRequest(headers=headers)
        )

    async def test__get_client_config__default_region_in_config__no_extra(self):
        request = self._get_request()

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "test-region"
        assert conf.claude_client == "legacy"
        assert conf.credentials is None

    async def test__get_client_config__default_region_in_config__empty_extra(
        self,
    ):
        request = self._get_request(extra_data={})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "test-region"
        assert conf.claude_client == "legacy"
        assert conf.credentials is None

    async def test__get_client_config__region_provided__region_in_config(self):
        request = self._get_request(extra_data={"region": "us-east-2"})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "us-east-2"
        assert conf.claude_client == "legacy"
        assert conf.credentials is None

    @pytest.mark.parametrize("claude_client", ["legacy", "mantle", "converse"])
    async def test__get_client_config__claude_client(self, claude_client):
        request = self._get_request(extra_data={"claude_client": claude_client})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "test-region"
        assert conf.claude_client == claude_client
        assert conf.credentials is None

    @pytest.mark.parametrize("default_client", ["legacy", "mantle", "converse"])
    async def test__get_client_config__default_client_from_env(
        self, monkeypatch, default_client
    ):
        monkeypatch.setenv("AWS_CLAUDE_DEFAULT_CLIENT", default_client)
        request = self._get_request(extra_data={"region": "us-east-2"})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "us-east-2"
        assert conf.claude_client == default_client

    async def test__get_client_config__header_client_overrides_env(
        self, monkeypatch
    ):
        monkeypatch.setenv("AWS_CLAUDE_DEFAULT_CLIENT", "mantle")
        request = self._get_request(extra_data={"claude_client": "legacy"})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.claude_client == "legacy"

    async def test__get_client_config__key_in_config(self):
        request = self._get_request(
            extra_data={
                "aws_access_key_id": "key_id",
                "aws_secret_access_key": "key",
            }
        )

        conf = await parse_upstream_config(request)  # type: ignore
        assert isinstance(conf, CloudUpstreamConfig)

        assert conf.region == "test-region"
        assert conf.claude_client == "legacy"

        _expiration, creds = await conf.get_credentials(session_tags=None)
        assert creds is not None
        assert creds.aws_access_key_id == "key_id"
        assert creds.aws_secret_access_key == "key"  # noqa: S105

    @mock.patch.object(
        AWSAssumeRoleCredentials,
        "get_credentials",
        return_value=(
            datetime.now(),
            ClientCredentialArgs(
                aws_access_key_id="key_id",
                aws_secret_access_key="key",  # noqa: S106
                aws_session_token="session_token",  # noqa: S106
            ),
        ),
    )
    async def test__get_client_config__role_arn__tmp_credentials_in_config(
        self, _mock
    ):
        request = self._get_request(extra_data={"aws_assume_role_arn": "arn"})

        conf = await parse_upstream_config(request)  # type: ignore
        assert isinstance(conf, CloudUpstreamConfig)

        assert conf.region == "test-region"
        assert conf.claude_client == "legacy"

        _expiration, creds = await conf.get_credentials(session_tags=None)
        assert creds is not None
        assert creds.aws_access_key_id == "key_id"
        assert creds.aws_secret_access_key == "key"  # noqa: S105
        assert creds.aws_session_token == "session_token"  # noqa: S105

    async def test_assume_role_passes_session_tags(self, monkeypatch):
        captured: dict = {}

        class _Sts:
            def assume_role(self, **kwargs):
                captured.update(kwargs)
                return {
                    "Credentials": {
                        "Expiration": datetime.now(),
                        "AccessKeyId": "a",
                        "SecretAccessKey": "s",
                        "SessionToken": "t",
                    }
                }

        class _Session:
            def client(self, *args, **kwargs):
                return _Sts()

        monkeypatch.setattr(
            "aidial_adapter_bedrock.upstream_config.boto3.Session",
            lambda: _Session(),
        )

        creds_config = AWSAssumeRoleCredentials(aws_assume_role_arn="arn")
        tags: list[SessionTag] = [{"Key": "UserInfo.roles.0", "Value": "admin"}]

        _expiration, creds = await creds_config.get_credentials(
            "us-east-1", tags
        )

        assert captured["RoleArn"] == "arn"
        assert captured["RoleSessionName"] == "BedrockAccessSession"
        assert captured["Tags"] == tags
        assert creds.aws_access_key_id == "a"

    async def test_assume_role_names_the_session_after_the_project(
        self, monkeypatch
    ):
        captured: dict = {}

        class _Sts:
            def assume_role(self, **kwargs):
                captured.update(kwargs)
                return {
                    "Credentials": {
                        "Expiration": datetime.now(),
                        "AccessKeyId": "a",
                        "SecretAccessKey": "s",
                        "SessionToken": "t",
                    }
                }

        class _Session:
            def client(self, *args, **kwargs):
                return _Sts()

        monkeypatch.setattr(
            "aidial_adapter_bedrock.upstream_config.boto3.Session",
            lambda: _Session(),
        )

        creds_config = AWSAssumeRoleCredentials(aws_assume_role_arn="arn")
        tags: list[SessionTag] = [{"Key": "UserInfo.project", "Value": "epam"}]

        await creds_config.get_credentials("us-east-1", tags)

        assert captured["RoleSessionName"] == "Project_epam"

    async def test_assume_role_omits_tags_when_empty(self, monkeypatch):
        captured: dict = {}

        class _Sts:
            def assume_role(self, **kwargs):
                captured.update(kwargs)
                return {
                    "Credentials": {
                        "Expiration": datetime.now(),
                        "AccessKeyId": "a",
                        "SecretAccessKey": "s",
                        "SessionToken": "t",
                    }
                }

        class _Session:
            def client(self, *args, **kwargs):
                return _Sts()

        monkeypatch.setattr(
            "aidial_adapter_bedrock.upstream_config.boto3.Session",
            lambda: _Session(),
        )

        creds_config = AWSAssumeRoleCredentials(aws_assume_role_arn="arn")

        await creds_config.get_credentials("us-east-1", None)

        assert "Tags" not in captured

    async def test__get_client_config__api_key_config(self):
        request = self._get_request(api_key="api-key")

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, ApiKeyUpstreamConfig)
        assert conf.api_key == "api-key"

    async def test__get_client_config__api_key_takes_precedence_over_client(
        self,
    ):
        request = self._get_request(
            extra_data={"claude_client": "invalid"},
            api_key="api-key",
        )

        conf = await parse_upstream_config(request)  # type: ignore
        assert isinstance(conf, ApiKeyUpstreamConfig)
        assert conf.api_key == "api-key"

    async def test__get_client_config__invalid_client_value(self):
        request = self._get_request(extra_data={"claude_client": "invalid"})

        with pytest.raises(ValidationError):
            await parse_upstream_config(request)  # type: ignore
