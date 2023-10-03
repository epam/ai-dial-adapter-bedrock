import argparse

from pydantic import BaseModel

# default values
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5001


class ClientConfig(BaseModel):
    host: str
    port: int
    n: int
    temperature: float
    streaming: bool


def get_client_config() -> ClientConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Server host, default is {DEFAULT_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server hort, default is {DEFAULT_PORT}",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        help="Number of messages to generate, default is 1",
    )
    parser.add_argument(
        "-t",
        type=float,
        default=0.0,
        help="Temperature, default is 0.0",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Streaming mode, default is False",
    )

    args = parser.parse_args()

    return ClientConfig(
        host=args.host,
        port=args.port,
        n=args.n,
        temperature=args.t,
        streaming=args.streaming,
    )
