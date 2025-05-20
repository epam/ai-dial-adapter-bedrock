import os
from datetime import datetime
from typing import Tuple

import boto3
from aidial_sdk.embeddings import Request
from pydantic import BaseModel, Field

from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.env import get_aws_default_region


class AWSClientCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str | None = None


class AWSAssumeRoleCredentials(BaseModel):
    aws_assume_role_arn: str

    async def get_tmp_credentials(
        self, region: str
    ) -> Tuple[datetime, AWSClientCredentials]:
        sts_client = await make_async(
            lambda: boto3.Session().client("sts", region_name=region)
        )

        response = sts_client.assume_role(
            RoleArn=self.aws_assume_role_arn,
            RoleSessionName="BedrockAccessSession",
        )

        creds = response["Credentials"]

        return creds["Expiration"], AWSClientCredentials(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )


class AWSClientConfig(BaseModel):
    region: str
    credentials: AWSClientCredentials | AWSAssumeRoleCredentials | None = None

    async def get_credentials(
        self,
    ) -> Tuple[datetime | None, AWSClientCredentials | None]:
        if self.credentials is None:
            return (None, None)
        if isinstance(self.credentials, AWSClientCredentials):
            return (None, self.credentials)
        return await self.credentials.get_tmp_credentials(self.region)


class UpstreamConfig(BaseModel):
    region: str = Field(default_factory=get_aws_default_region)
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_assume_role_arn: str | None = os.getenv("AWS_ASSUME_ROLE_ARN")


class AWSClientConfigFactory:
    UPSTREAM_CONFIG_HEADER_NAME = "x-upstream-extra-data"

    def __init__(self, request):
        self.upstream_config = self._get_upstream_config(request)

    def get_client_config(self) -> AWSClientConfig:
        return AWSClientConfig(
            region=self.upstream_config.region,
            credentials=self._get_client_credentials(),
        )

    def _get_upstream_config(self, request: Request) -> UpstreamConfig:
        conf = request.headers.get(self.UPSTREAM_CONFIG_HEADER_NAME)
        return UpstreamConfig.parse_raw(conf) if conf else UpstreamConfig()

    def _get_client_credentials(
        self,
    ) -> AWSClientCredentials | AWSAssumeRoleCredentials | None:

        conf = self.upstream_config

        if conf.aws_access_key_id and conf.aws_secret_access_key:
            return AWSClientCredentials(
                aws_access_key_id=conf.aws_access_key_id,
                aws_secret_access_key=conf.aws_secret_access_key,
            )

        if conf.aws_assume_role_arn:
            return AWSAssumeRoleCredentials(
                aws_assume_role_arn=conf.aws_assume_role_arn
            )

        return None
