from dataclasses import dataclass
from datetime import datetime
from unittest import mock

from aidial_adapter_bedrock.aws_client_config import (
    AWSAssumeRoleCredentials,
    AWSClientConfigFactory,
    AWSClientCredentials,
)


@dataclass
class FakeRequest:
    headers: dict[str, str]


class TestAWSClientConfigFactory:
    @staticmethod
    def _get_request(raw_upstream_config):
        header_name = AWSClientConfigFactory.UPSTREAM_CONFIG_HEADER_NAME
        return FakeRequest(headers={header_name: raw_upstream_config})

    async def test__get_client_config__default_region_in_config(self):
        request = FakeRequest(headers={})

        client_config = AWSClientConfigFactory(request).get_client_config()

        assert client_config.region == "test-region"
        assert client_config.credentials is None

    async def test__get_client_config__region_provided__region_in_config(self):
        raw_upstream_config = '{"region": "us-east-2"}'
        request = self._get_request(raw_upstream_config)

        client_config = AWSClientConfigFactory(request).get_client_config()

        assert client_config.region == "us-east-2"
        assert client_config.credentials is None

    async def test__get_client_config__key_in_config(self):
        raw_upstream_config = (
            '{"aws_access_key_id": "key_id", "aws_secret_access_key": "key"}'
        )
        request = self._get_request(raw_upstream_config)

        client_config = AWSClientConfigFactory(request).get_client_config()
        assert client_config.region == "test-region"

        (_expiration, creds) = await client_config.get_credentials()
        assert creds is not None
        assert creds.aws_access_key_id == "key_id"
        assert creds.aws_secret_access_key == "key"

    @mock.patch.object(
        AWSAssumeRoleCredentials,
        "get_tmp_credentials",
        return_value=(
            datetime.now(),
            AWSClientCredentials(
                aws_access_key_id="key_id",
                aws_secret_access_key="key",
                aws_session_token="session_token",
            ),
        ),
    )
    async def test__get_client_config__role_arn__tmp_credentials_in_config(
        self, _mock
    ):
        raw_upstream_config = '{"aws_assume_role_arn": "arn"}'
        request = self._get_request(raw_upstream_config)

        client_config = AWSClientConfigFactory(request).get_client_config()

        assert client_config.region == "test-region"

        (_expiration, creds) = await client_config.get_credentials()
        assert creds is not None
        assert creds.aws_access_key_id == "key_id"
        assert creds.aws_secret_access_key == "key"
        assert creds.aws_session_token == "session_token"
