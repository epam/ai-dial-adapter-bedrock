import asyncio
from concurrent.futures import ThreadPoolExecutor


async def make_async(func, *args):
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)
