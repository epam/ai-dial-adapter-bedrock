"""
Deployment capabilities, read off the inbound request.

DIAL Core fronts several model adapters whose accepted request shapes genuinely
differ, and the difference is not inferable from the request body: a Gemini
deployment configured with a thinking budget rejects any request that also
carries `reasoning_effort`, because the Vertex adapter maps that onto Gemini's
`thinking_level`. A translator that emits one fixed shape therefore works
against some deployments and hard-fails against others.

Core stamps `x-dial-deployment-features` on every request it routes here: the
`features` object it publishes for the target deployment in its model listing,
already scoped to the caller's roles and to the deployment Core resolved.
Reading it costs no HTTP call and no cache.

A header is a declaration, not a proof — one has been seen reporting
`"tools": false` for a deployment that plainly supports tools — so each flag is
read only for the field it governs. A header that never arrived, or one that
cannot be read, yields `UNRESOLVED_PROFILE`, which means "unknown", not
"unsupported": every gate goes quiet, because sending nothing degrades a
request while sending the wrong knob fails it.
"""

import json
from typing import Any

from pydantic import BaseModel, StrictBool, StrictStr, ValidationError
from starlette.datastructures import Headers

from aidial_adapter_bedrock.utils.log_config import bedrock_logger as log
from aidial_adapter_bedrock.utils.pydantic import ExtraAllowModel

FEATURES_HEADER = "x-dial-deployment-features"


class DeploymentFeatures(ExtraAllowModel):
    """The `features` object as Core publishes it.

    Every flag arrives as a JSON boolean, so it is read as one: a
    non-boolean is malformed rather than truthy, and the profile it produces
    resolves the safe way. Flags outside these four change nothing here.
    """

    temperature: StrictBool | None = None
    cache: StrictBool | None = None
    reasoning_efforts: list[StrictStr] | None = None
    max_completion_tokens_supported: StrictBool | None = None


class DeploymentProfile(BaseModel):
    """The request shape one deployment accepts."""

    temperature_supported: bool
    cache_supported: bool
    max_completion_tokens_supported: bool
    reasoning_efforts: list[str]


def _to_profile(features: DeploymentFeatures) -> DeploymentProfile:
    """Resolve the advertised features into gates.

    Gate direction is chosen per field by what the wrong answer costs, which is
    why the gates don't point the same way: dropping `temperature` on a guess
    silently changes generation, so only an explicit `false` suppresses it,
    while every other gate stays shut until the header opens it, because
    emitting a field the adapter doesn't take fails the request outright.
    """
    return DeploymentProfile(
        temperature_supported=features.temperature is not False,
        cache_supported=features.cache is True,
        max_completion_tokens_supported=features.max_completion_tokens_supported
        is True,
        reasoning_efforts=features.reasoning_efforts or [],
    )


UNRESOLVED_PROFILE = _to_profile(DeploymentFeatures())


def parse_deployment_profile(headers: Headers) -> DeploymentProfile:
    """The profile the `x-dial-deployment-features` header declares.

    Never raises: a call that did not come through Core carries no header at
    all, and a capability lookup must never fail the user's message.
    """
    raw: str | None = headers.get(FEATURES_HEADER)
    if not raw:
        log.debug(
            "No %s header; deployment capabilities unknown", FEATURES_HEADER
        )
        return UNRESOLVED_PROFILE

    try:
        features: Any = json.loads(raw)
        if not isinstance(features, dict):
            raise ValueError("not a JSON object")
        return _to_profile(DeploymentFeatures.model_validate(features))
    except (ValueError, ValidationError):
        log.warning(
            "Could not read the %s header; continuing without deployment "
            "capabilities",
            FEATURES_HEADER,
            exc_info=True,
        )
        return UNRESOLVED_PROFILE
