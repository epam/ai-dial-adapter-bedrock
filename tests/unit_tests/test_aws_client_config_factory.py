import json
from dataclasses import dataclass
from datetime import datetime
from unittest import mock

from aidial_adapter_bedrock.upstream_config import (
    ApiKeyUpstreamConfig,
    AWSAssumeRoleCredentials,
    ClientCredentialArgs,
    CloudUpstreamConfig,
    parse_upstream_config,
)


@dataclass
class FakeRequest:
    headers: dict[str, str]


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
        return FakeRequest(headers=headers)

    async def test__get_client_config__default_region_in_config__no_extra(self):
        request = self._get_request()

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "test-region"
        assert conf.credentials is None

    async def test__get_client_config__default_region_in_config__empty_extra(
        self,
    ):
        request = self._get_request(extra_data={})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "test-region"
        assert conf.credentials is None

    async def test__get_client_config__region_provided__region_in_config(self):
        request = self._get_request(extra_data={"region": "us-east-2"})

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, CloudUpstreamConfig)
        assert conf.region == "us-east-2"
        assert conf.credentials is None

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

        (_expiration, creds) = await conf.get_credentials()
        assert creds is not None
        assert creds.aws_access_key_id == "key_id"
        assert creds.aws_secret_access_key == "key"

    @mock.patch.object(
        AWSAssumeRoleCredentials,
        "get_tmp_credentials",
        return_value=(
            datetime.now(),
            ClientCredentialArgs(
                aws_access_key_id="key_id",
                aws_secret_access_key="key",
                aws_session_token="session_token",
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

        (_expiration, creds) = await conf.get_credentials()
        assert creds is not None
        assert creds.aws_access_key_id == "key_id"
        assert creds.aws_secret_access_key == "key"
        assert creds.aws_session_token == "session_token"

    async def test__get_client_config__api_key_config(self):
        request = self._get_request(api_key="api-key")

        conf = await parse_upstream_config(request)  # type: ignore

        assert isinstance(conf, ApiKeyUpstreamConfig)
        assert conf.api_key == "api-key"
