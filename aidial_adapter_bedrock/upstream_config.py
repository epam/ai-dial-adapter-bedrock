import os
from datetime import datetime
from typing import ClassVar, Optional, TypedDict, assert_never

import boto3
import fastapi
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.env import (
    AWSClaudeClient,
    get_aws_default_region,
    get_default_claude_client,
)
from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log

_UPSTREAM_CONFIG_HEADER_NAME = "x-upstream-extra-data"


class SessionTag(TypedDict):
    Key: str
    Value: str


class ClientCredentialArgs(BaseModel):
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None


class AWSClientCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str | None = None

    def get_credentials(self) -> tuple[datetime | None, ClientCredentialArgs]:
        return None, ClientCredentialArgs(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )


class AWSAssumeRoleCredentials(BaseModel):
    aws_assume_role_arn: str

    async def get_credentials(
        self,
        region: str,
        session_tags: list[SessionTag] | None = None,
    ) -> tuple[datetime, ClientCredentialArgs]:
        sts_client = await make_async(
            lambda: boto3.Session().client("sts", region_name=region)
        )

        assume_role_params: dict = {
            "RoleArn": self.aws_assume_role_arn,
            "RoleSessionName": "BedrockAccessSession",
        }
        if session_tags:
            assume_role_params["Tags"] = session_tags

        response = sts_client.assume_role(**assume_role_params)

        creds = response["Credentials"]

        return creds["Expiration"], ClientCredentialArgs(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )


class CloudUpstreamConfig(BaseModel):
    region: str
    credentials: AWSClientCredentials | AWSAssumeRoleCredentials | None = None
    claude_client: AWSClaudeClient

    @classmethod
    async def from_request(
        cls, request: fastapi.Request
    ) -> "CloudUpstreamConfig":
        conf = request.headers.get(_UPSTREAM_CONFIG_HEADER_NAME)
        upstream_config = (
            UpstreamConfigData.model_validate_json(conf)
            if conf
            else UpstreamConfigData()
        )

        return cls(
            region=upstream_config.region,
            credentials=upstream_config._get_client_credentials(),
            claude_client=upstream_config.claude_client,
        )

    async def get_credentials(
        self, session_tags: list[SessionTag] | None
    ) -> tuple[datetime | None, ClientCredentialArgs]:
        match self.credentials:
            case None:
                return None, ClientCredentialArgs()
            case AWSClientCredentials():
                return self.credentials.get_credentials()
            case AWSAssumeRoleCredentials():
                return await self.credentials.get_credentials(
                    self.region, session_tags
                )
            case _:
                assert_never(self.credentials)


class ApiKeyUpstreamConfig(BaseModel):
    _UPSTREAM_API_KEY_HEADER_NAME: ClassVar[str] = "x-upstream-key"

    api_key: str

    @classmethod
    def from_request(
        cls, request: fastapi.Request
    ) -> Optional["ApiKeyUpstreamConfig"]:
        key = request.headers.get(cls._UPSTREAM_API_KEY_HEADER_NAME)
        return None if key is None else cls(api_key=key)


UpstreamConfig = ApiKeyUpstreamConfig | CloudUpstreamConfig


async def parse_upstream_config(request: fastapi.Request) -> UpstreamConfig:
    if (conf := ApiKeyUpstreamConfig.from_request(request)) is not None:
        log.debug("accessing deployment via platform api-key")
        return conf

    log.debug("accessing deployment via cloud creds")
    return await CloudUpstreamConfig.from_request(request)


class UpstreamConfigData(BaseModel):
    region: str = Field(default_factory=get_aws_default_region)
    claude_client: AWSClaudeClient = Field(
        default_factory=get_default_claude_client
    )
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token: str | None = os.getenv("AWS_SESSION_TOKEN")
    aws_assume_role_arn: str | None = os.getenv("AWS_ASSUME_ROLE_ARN")

    def _get_client_credentials(
        self,
    ) -> AWSClientCredentials | AWSAssumeRoleCredentials | None:
        if self.aws_access_key_id and self.aws_secret_access_key:
            return AWSClientCredentials(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
            )

        if self.aws_assume_role_arn:
            return AWSAssumeRoleCredentials(
                aws_assume_role_arn=self.aws_assume_role_arn
            )

        return None


class OverrideNameUpstreamConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    compatible_model_id: str | None = None


def get_compatible_model_id(request: fastapi.Request) -> str | None:
    if (extra := request.headers.get(_UPSTREAM_CONFIG_HEADER_NAME)) is None:
        return None

    try:
        conf = OverrideNameUpstreamConfig.model_validate_json(extra)
    except Exception as e:
        log.error(
            f"Request header {_UPSTREAM_CONFIG_HEADER_NAME!r} doesn't contain"
            f" valid override name configuration: {e}"
        )
        return None

    return None if conf is None else conf.compatible_model_id
