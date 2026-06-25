import json
from dataclasses import dataclass
from datetime import datetime
from unittest import mock

import pytest
from aidial_adapter_anthropic.adapter import ValidationError

from aidial_adapter_bedrock.upstream_config import (
    ApiKeyUpstreamConfig,
    AWSAssumeRoleCredentials,
    ClientCredentialArgs,
    CloudUpstreamConfig,
    parse_upstream_config,
)


@dataclass
class OriginalRequest:
    headers: dict[str, str]


@dataclass
class FakeRequest:
    headers: dict[str, str]
    original_request: OriginalRequest


class TestAWSClientConfigFactory:
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
        assert conf.client == "legacy"
        assert conf.credentials is None

    async def test__get_client_config__region_provided__region_in_config(self):
        request = self._get_request(extra_data={"region": "us-east-2"})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "us-east-2"
        assert conf.client == "legacy"
        assert conf.credentials is None

    async def test__get_client_config__mantle_client(self):
        request = self._get_request(extra_data={"client": "mantle"})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "test-region"
        assert conf.client == "mantle"
        assert conf.credentials is None

    async def test__get_client_config__header_client_overrides_env(
        self, monkeypatch
    ):
        monkeypatch.setenv("AWS_DEFAULT_CLIENT", "mantle")
        request = self._get_request(extra_data={"client": "legacy"})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.client == "legacy"

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
        assert conf.client == "legacy"

        _expiration, creds = await conf.get_credentials()
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
        assert conf.client == "legacy"

        _expiration, creds = await conf.get_credentials()
        assert creds is not None
        assert creds.aws_access_key_id == "key_id"
        assert creds.aws_secret_access_key == "key"  # noqa: S105
        assert creds.aws_session_token == "session_token"  # noqa: S105

    async def test__get_client_config__api_key_config(self):
        request = self._get_request(api_key="api-key")

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, ApiKeyUpstreamConfig)
        assert conf.api_key == "api-key"

    async def test__get_client_config__api_key_takes_precedence_over_client(
        self,
    ):
        request = self._get_request(
            extra_data={"client": "invalid"},
            api_key="api-key",
        )

        conf = await parse_upstream_config(request)  # type: ignore
        assert isinstance(conf, ApiKeyUpstreamConfig)
        assert conf.api_key == "api-key"

    async def test__get_client_config__invalid_client_value(self):
        request = self._get_request(extra_data={"client": "invalid"})

        with pytest.raises(ValidationError) as e:
            await parse_upstream_config(request)  # type: ignore

        assert "x-upstream-extra-data" in str(e.value)
        assert "client" in str(e.value)
        assert "legacy" in str(e.value)
        assert "mantle" in str(e.value)
