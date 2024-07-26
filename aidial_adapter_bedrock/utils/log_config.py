import logging
import os
import sys

from aidial_sdk import logger as aidial_logger
from uvicorn.logging import DefaultFormatter

# By default (in prod) we don't want to print debug messages,
# because they typically contain prompts.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AIDIAL_LOG_LEVEL = os.getenv("AIDIAL_LOG_LEVEL", "WARNING")
aidial_logger.setLevel(AIDIAL_LOG_LEVEL)


def configure_loggers():
    # Making the uvicorn and dial_sdk loggers delegate its logging to the root logger
    for logger in [logging.getLogger("uvicorn"), aidial_logger]:
        logger.handlers = []
        logger.propagate = True

    # Setting up log levels
    for name in ["app", "bedrock", "uvicorn", "__main__"]:
        logging.getLogger(name).setLevel(LOG_LEVEL)

    # Configuring the root logger
    root = logging.getLogger()

    root_has_stderr_handler = any(
        isinstance(handler, logging.StreamHandler)
        and handler.stream == sys.stderr
        for handler in root.handlers
    )

    if not root_has_stderr_handler:
        formatter = DefaultFormatter(
            fmt="%(levelprefix)s | %(asctime)s | %(process)d | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_colors=True,
        )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)


# Loggers in order from high-level to low-level
# High-level logs from the adapter server
app_logger = logging.getLogger("app")

# LLM requests and responses
bedrock_logger = logging.getLogger("bedrock")
