import os
from typing import Any

from aidial_client import UserInfo

CONVERSE_API_REQUEST_METADATA_FIELDS = os.getenv(
    "CONVERSE_API_REQUEST_METADATA_FIELDS"
)


def request_metadata_fields() -> dict[str, str]:
    if CONVERSE_API_REQUEST_METADATA_FIELDS is None:
        return {}

    fields = CONVERSE_API_REQUEST_METADATA_FIELDS.strip()
    if not fields:
        return {}

    available_fields = tuple(UserInfo.model_fields)
    available_fields_set = set(UserInfo.model_fields)

    if fields == "*":
        return dict.fromkeys(available_fields, "")

    requested_fields = [field.strip() for field in fields.split(",")]
    unknown_fields = set(requested_fields) - available_fields_set
    if unknown_fields:
        raise ValueError(
            "Unknown UserInfo fields: " + ", ".join(sorted(unknown_fields))
        )

    return dict.fromkeys(requested_fields, "")


def from_user_info(user_info: UserInfo) -> dict[str, Any]:
    return user_info.model_dump(include=set(request_metadata_fields()))
