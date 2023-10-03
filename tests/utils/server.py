import time
from multiprocessing import Process
from urllib.parse import urlparse

import requests
import uvicorn


def ping_server(url: str) -> bool:
    try:
        requests.get(url, timeout=1)
        return True
    except requests.ConnectionError:
        return False


def wait_for_server(url: str, timeout=10) -> None:
    start_time = time.time()

    while True:
        if ping_server(url):
            return

        if time.time() - start_time > timeout:
            raise Exception("The test server didn't start in time!")

        time.sleep(0.1)


def terminate_process(process: Process):
    process.terminate()
    process.join()


def server_generator(module: str, url: str):
    already_exists = ping_server(url)

    server_process: Process | None = None
    if not already_exists:
        parsed_url = urlparse(url)
        server_process = Process(
            target=uvicorn.run,
            args=(module,),
            kwargs={"host": parsed_url.hostname, "port": parsed_url.port},
        )
        server_process.start()

        try:
            wait_for_server(url)
        except Exception as e:
            terminate_process(server_process)
            raise Exception("Can't start the test server") from e

    yield

    if server_process is not None:
        terminate_process(server_process)
