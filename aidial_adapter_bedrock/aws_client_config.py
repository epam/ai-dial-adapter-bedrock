import json

import boto3
from aidial_sdk.embeddings import Request
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.env import get_aws_default_region


class AWSClientCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str | None = None


class AWSClientConfig(BaseModel):
    region: str
    credentials: AWSClientCredentials | None = None

    def get_boto_client_kwargs(self) -> dict:
        client_kwargs = {"region_name": self.region}

        if self.credentials:
            client_kwargs.update(self.credentials.dict(exclude_none=True))

        return client_kwargs

    def get_anthropic_bedrock_client_kwargs(self) -> dict:
        client_kwargs = {"aws_region": self.region}
        if self.credentials:
            if self.credentials.aws_access_key_id:
                client_kwargs.update(
                    {"aws_access_key": self.credentials.aws_access_key_id}
                )

            if self.credentials.aws_secret_access_key:
                client_kwargs.update(
                    {"aws_secret_key": self.credentials.aws_secret_access_key}
                )

            if self.credentials.aws_session_token:
                client_kwargs.update(
                    {"aws_session_token": self.credentials.aws_session_token}
                )

        return client_kwargs


class AWSClientConfigFactory:
    RAW_UPSTREAM_CONFIG_HEADER_NAME = "x-upstream-extra-data"
    BEDROCK_ACCESS_SESSION_NAME = "BedrockAccessSession"

    def __init__(self, request):
        upstream_config = self._get_upstream_config_from_request(request)

        self._region = upstream_config.get("region", get_aws_default_region())
        self._access_key_id = upstream_config.get("aws_access_key_id")
        self._secret_access_key = upstream_config.get("aws_secret_access_key")
        self._assumed_role_arn = upstream_config.get("aws_assume_role_arn")

    async def get_client_config(self) -> AWSClientConfig:
        return AWSClientConfig(
            region=self._region,
            credentials=await self._get_client_credentials(),
        )

    def _get_upstream_config_from_request(self, request: Request) -> dict:
        raw_upstream_config = request.headers.get(
            self.RAW_UPSTREAM_CONFIG_HEADER_NAME
        )
        return json.loads(raw_upstream_config) if raw_upstream_config else {}

    async def _get_client_credentials(self) -> AWSClientCredentials | None:
        if self._access_key_id and self._secret_access_key:
            return AWSClientCredentials(
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
            )

        if self._assumed_role_arn:
            return await self._get_assumed_role_tmp_credentials()

    async def _get_assumed_role_tmp_credentials(self) -> AWSClientCredentials:
        sts_client = await make_async(
            lambda: boto3.Session().client("sts", region_name=self._region)
        )

        assumed_role_object = sts_client.assume_role(
            RoleArn=self._assumed_role_arn,
            RoleSessionName=self.BEDROCK_ACCESS_SESSION_NAME,
        )

        return AWSClientCredentials(
            aws_access_key_id=assumed_role_object["Credentials"]["AccessKeyId"],
            aws_secret_access_key=assumed_role_object["Credentials"][
                "SecretAccessKey"
            ],
            aws_session_token=assumed_role_object["Credentials"][
                "SessionToken"
            ],
        )
