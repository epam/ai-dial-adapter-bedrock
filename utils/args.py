import argparse

# default values
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


def get_host_port_args():
    parser = argparse.ArgumentParser(description="Run the app")

    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host to run the app on, default is {DEFAULT_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the app on, default is {DEFAULT_PORT}",
    )

    args = parser.parse_args()

    return args.host, args.port