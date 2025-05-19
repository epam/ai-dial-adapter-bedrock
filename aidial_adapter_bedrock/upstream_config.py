from __future__ import annotations

import os
from functools import cache
from typing import Any, ClassVar

import anthropic
import boto3
import httpx
from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from pydantic import BaseModel, Field

from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.env import get_aws_default_region
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log


@cache
def _get_default_anthropic_timeout() -> httpx.Timeout:
    # Providing a timeout marginally different from the default Anthropic timeout
    # in order to disable the check that throws an error when
    # stream=False & max_tokens>=128K/6:
    # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176

    timeout = anthropic._constants.DEFAULT_TIMEOUT.as_dict()
    timeout["connect"] *= 1.0001  # type: ignore
    return httpx.Timeout(**timeout)


class ApiKeyUpstreamConfig(BaseModel):
    _UPSTREAM_API_KEY_HEADER_NAME: ClassVar[str] = "x-upstream-key"

    api_key: str

    @classmethod
    def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> ApiKeyUpstreamConfig | None:
        key = request.headers.get(cls._UPSTREAM_API_KEY_HEADER_NAME)
        return None if key is None else cls(api_key=key)

    def get_anthropic_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(
            api_key=self.api_key,
            timeout=_get_default_anthropic_timeout(),
        )


class AWSClientCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str | None


class CloudUpstreamConfig(BaseModel):
    _UPSTREAM_CONFIG_HEADER_NAME: ClassVar[str] = "x-upstream-extra-data"

    region: str
    credentials: AWSClientCredentials | None = None

    @classmethod
    async def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> CloudUpstreamConfig:
        conf = request.headers.get(cls._UPSTREAM_CONFIG_HEADER_NAME)
        upstream_config = (
            UpstreamConfigData.parse_raw(conf) if conf else UpstreamConfigData()
        )

        return cls(
            region=upstream_config.region,
            credentials=await upstream_config._get_client_credentials(),
        )

    async def get_bedrock_client(self) -> Any:
        creds = self.credentials
        return await make_async(
            lambda: boto3.Session().client(
                service_name="bedrock-runtime",
                region_name=self.region,
                aws_access_key_id=creds and creds.aws_access_key_id,
                aws_secret_access_key=creds and creds.aws_secret_access_key,
                aws_session_token=creds and creds.aws_session_token,
            )
        )

    def get_anthropic_client(self) -> AsyncAnthropicBedrock:
        creds = self.credentials
        return AsyncAnthropicBedrock(
            aws_region=self.region,
            aws_access_key=creds and creds.aws_access_key_id,
            aws_secret_key=creds and creds.aws_secret_access_key,
            aws_session_token=creds and creds.aws_session_token,
            timeout=_get_default_anthropic_timeout(),
        )


UpstreamConfig = ApiKeyUpstreamConfig | CloudUpstreamConfig


def to_cloud_config(conf: UpstreamConfig) -> CloudUpstreamConfig:
    if isinstance(conf, ApiKeyUpstreamConfig):
        raise ValueError(
            "Authentication via API key isn't supported for the deployment"
        )
    return conf


async def parse_upstream_config(
    request: FromRequestDeploymentMixin,
) -> UpstreamConfig:
    if (conf := ApiKeyUpstreamConfig.from_request(request)) is not None:
        log.debug("accessing deployment via platform api-key")
        return conf

    log.debug("accessing deployment via cloud creds")
    return await CloudUpstreamConfig.from_request(request)


_BEDROCK_ACCESS_SESSION_NAME = "BedrockAccessSession"


class UpstreamConfigData(BaseModel):
    region: str = Field(default_factory=get_aws_default_region)
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_assume_role_arn: str | None = os.getenv("AWS_ASSUME_ROLE_ARN")

    async def _get_client_credentials(self) -> AWSClientCredentials | None:
        key_id = self.aws_access_key_id
        secret_access_key = self.aws_secret_access_key

        if key_id and secret_access_key:
            return AWSClientCredentials(
                aws_access_key_id=key_id,
                aws_secret_access_key=secret_access_key,
                aws_session_token=None,
            )

        if role_arn := self.aws_assume_role_arn:
            return await UpstreamConfigData._get_assumed_role_tmp_credentials(
                self.region, role_arn
            )

        return None

    @staticmethod
    async def _get_assumed_role_tmp_credentials(
        region: str, role_arn: str
    ) -> AWSClientCredentials:
        sts_client = await make_async(
            lambda: boto3.Session().client("sts", region_name=region)
        )

        assumed_role_object = sts_client.assume_role(
            RoleArn=role_arn, RoleSessionName=_BEDROCK_ACCESS_SESSION_NAME
        )

        assumed_role_credentials = assumed_role_object["Credentials"]

        return AWSClientCredentials(
            aws_access_key_id=assumed_role_credentials["AccessKeyId"],
            aws_secret_access_key=assumed_role_credentials["SecretAccessKey"],
            aws_session_token=assumed_role_credentials["SessionToken"],
        )
