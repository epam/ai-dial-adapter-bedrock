import threading
from typing import Any

import boto3

# A single shared Session, so that every client reuses the botocore loader and
# its cache of parsed service models. A client built from a throwaway
# `boto3.Session()` retains ~8MB of service models of its own, while one built
# from a shared Session costs ~0.5MB.
#
# NOTE: Session isn't thread-safe, but client is, and clients are created on
# worker threads (see `make_async`), hence the lock.
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html#caveats
_session = boto3.Session()
_session_lock = threading.Lock()


def create_client(service_name: str, **kwargs: Any) -> Any:
    with _session_lock:
        return _session.client(service_name, **kwargs)
