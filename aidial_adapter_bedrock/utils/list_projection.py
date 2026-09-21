# The projection type `AdapterRequest.messages` is expressed in, re-exported
# so that the private import of the passthrough library lives in one place.
from aidial_adapter_anthropic._utils.list import ListProjection

__all__ = ["ListProjection"]
