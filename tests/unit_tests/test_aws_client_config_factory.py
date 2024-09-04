from dataclasses import dataclass
from unittest import mock

import pytest

from aidial_adapter_bedrock.aws_client_config import (
    AWSClientConfigFactory,
    AWSClientCredentials,
)


@dataclass
class FakeRequest:
    headers: dict[str, str]


@pytest.mark.asyncio
class TestAWSClientConfigFactory:
    async def test__get_client_config__no_raw_upstream_config__default_region_in_config(
        self,
    ):
        request = FakeRequest(headers={})

        client_config = await AWSClientConfigFactory(
            request=request
        ).get_client_config()

        assert client_config.region == "us-east-1"
        assert client_config.credentials is None

    async def test__get_client_config__region_provided__region_in_config(self):
        raw_upstream_config = '{"region": "us-east-2"}'
        request = FakeRequest(headers={AWSClientConfigFactory.RAW_UPSTREAM_CONFIG_HEADER_NAME: raw_upstream_config})

        client_config = await AWSClientConfigFactory(
            request=request,
        ).get_client_config()

        assert client_config.region == "us-east-2"
        assert client_config.credentials is None

    async def test__get_client_config__raw_upstream_config_with_key_credentials__key_in_config(
        self,
    ):
        raw_upstream_config = '{"aws_access_key_id": "key_id", "aws_secret_access_key": "key"}'
        request = FakeRequest(headers={AWSClientConfigFactory.RAW_UPSTREAM_CONFIG_HEADER_NAME: raw_upstream_config})

        client_config = await AWSClientConfigFactory(
            request=request,
        ).get_client_config()

        assert client_config.region == "us-east-1"
        assert client_config.credentials.aws_access_key_id == "key_id"
        assert client_config.credentials.aws_secret_access_key == "key"

    @mock.patch.object(
        AWSClientConfigFactory,
        "_get_assumed_role_tmp_credentials",
        return_value=AWSClientCredentials(
            aws_access_key_id="key_id",
            aws_secret_access_key="key",
            aws_session_token="session_token",
        ),
    )
    async def test__get_client_config__raw_upstream_config_with_role_arn__tmp_credentials_in_config(
        self, _mock
    ):
        raw_upstream_config = '{"aws_assume_role_arn": "arn"}'
        request = FakeRequest(headers={AWSClientConfigFactory.RAW_UPSTREAM_CONFIG_HEADER_NAME: raw_upstream_config})

        client_config = await AWSClientConfigFactory(
            request=request,
        ).get_client_config()

        assert client_config.region == "us-east-1"
        assert client_config.credentials.aws_access_key_id == "key_id"
        assert client_config.credentials.aws_secret_access_key == "key"
        assert client_config.credentials.aws_session_token == "session_token"
