import asyncio
import time

import pytest

from aidial_adapter_bedrock.utils.concurrency import (
    run_in_threadpool,
    to_async_iterator,
)


@pytest.mark.asyncio
async def test_run_in_threadpool_runs_in_a_thread():
    assert await run_in_threadpool(lambda: 42) == 42


@pytest.mark.asyncio
async def test_run_in_threadpool_propagates_errors():
    with pytest.raises(ValueError, match="boom"):
        await run_in_threadpool(
            lambda: (_ for _ in ()).throw(ValueError("boom"))
        )


@pytest.mark.asyncio
async def test_to_async_iterator():
    assert [x async for x in to_async_iterator(iter([1, 2, 3]))] == [1, 2, 3]


@pytest.mark.asyncio
async def test_cancellation_does_not_block_the_event_loop():
    """
    A `run_in_threadpool` call cancelled while its blocking function is still
    running must not stall the event loop: a pool per call used to join the
    worker on the loop thread, freezing /health for as long as the call took
    to time out.
    """
    blocking = 2.0
    task = asyncio.create_task(run_in_threadpool(lambda: time.sleep(blocking)))
    await asyncio.sleep(0.1)

    started = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The loop must keep turning while the abandoned thread runs on.
    await asyncio.sleep(0)
    elapsed = time.perf_counter() - started

    assert elapsed < blocking / 4, (
        f"the event loop was blocked for {elapsed:.2f}s by a cancelled call"
    )
