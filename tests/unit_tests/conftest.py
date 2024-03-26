import os


def pytest_configure(config):
    """
    Setting up fake environment variables for unit tests.
    """
    os.environ["AWS_DEFAULT_REGION"] = "dummy_region"
