import boto3
from aidial_sdk.embeddings import Request
from pydantic import BaseModel

from aidial_adapter_bedrock.utils.concurrency import make_async
from aidial_adapter_bedrock.utils.env import get_aws_default_region
from aidial_adapter_bedrock.utils.json import remove_nones


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
            credentials = remove_nones(
                {
                    "aws_access_key": self.credentials.aws_access_key_id,
                    "aws_secret_key": self.credentials.aws_secret_access_key,
                    "aws_session_token": self.credentials.aws_session_token,
                }
            )
            client_kwargs.update(credentials)
        return client_kwargs


class UpstreamConfig(BaseModel):
    region: str = get_aws_default_region()
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_assume_role_arn: str | None = None


class AWSClientConfigFactory:
    UPSTREAM_CONFIG_HEADER_NAME = "x-upstream-extra-data"
    BEDROCK_ACCESS_SESSION_NAME = "BedrockAccessSession"

    def __init__(self, request):
        self.upstream_config = self._get_upstream_config(request)

    async def get_client_config(self) -> AWSClientConfig:
        return AWSClientConfig(
            region=self.upstream_config.region,
            credentials=await self._get_client_credentials(),
        )

    def _get_upstream_config(self, request: Request) -> UpstreamConfig:
        conf = request.headers.get(self.UPSTREAM_CONFIG_HEADER_NAME)
        return UpstreamConfig.parse_raw(conf) if conf else UpstreamConfig()

    async def _get_client_credentials(self) -> AWSClientCredentials | None:
        key_id = self.upstream_config.aws_access_key_id
        secret_access_key = self.upstream_config.aws_secret_access_key

        if key_id and secret_access_key:
            return AWSClientCredentials(
                aws_access_key_id=key_id,
                aws_secret_access_key=secret_access_key,
            )

        if self.upstream_config.aws_assume_role_arn:
            return await self._get_assumed_role_tmp_credentials()

    async def _get_assumed_role_tmp_credentials(self) -> AWSClientCredentials:
        sts_client = await make_async(
            lambda: boto3.Session().client(
                "sts", region_name=self.upstream_config.region
            )
        )

        assumed_role_object = sts_client.assume_role(
            RoleArn=self.upstream_config.aws_assume_role_arn,
            RoleSessionName=self.BEDROCK_ACCESS_SESSION_NAME,
        )

        assumed_role_credentials = assumed_role_object["Credentials"]

        return AWSClientCredentials(
            aws_access_key_id=assumed_role_credentials["AccessKeyId"],
            aws_secret_access_key=assumed_role_credentials["SecretAccessKey"],
            aws_session_token=assumed_role_credentials["SessionToken"],
        )
