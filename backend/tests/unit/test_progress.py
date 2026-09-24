"""Tests for job progress fan-out."""

import asyncio

from orbital_recon.schemas.enums import JobStage, JobStatus
from orbital_recon.services.progress import ProgressTracker, ProgressUpdate


def update(
    job_id: int = 1,
    progress: float = 0.0,
    status: JobStatus = JobStatus.RUNNING,
    stage: JobStage = JobStage.INFERENCE,
) -> ProgressUpdate:
    return ProgressUpdate(job_id=job_id, status=status, stage=stage, progress=progress)


def terminal(job_id: int = 1, count: int = 0) -> ProgressUpdate:
    return ProgressUpdate(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        stage=JobStage.DONE,
        progress=1.0,
        detection_count=count,
    )


class TestProgressUpdate:
    def test_running_is_not_terminal(self) -> None:
        assert not update().is_terminal

    def test_completed_and_failed_are_terminal(self) -> None:
        assert terminal().is_terminal
        assert ProgressUpdate(
            job_id=1, status=JobStatus.FAILED, stage=JobStage.INFERENCE, progress=0.4
        ).is_terminal


class TestProgressTracker:
    async def test_subscriber_receives_updates_in_order(self) -> None:
        tracker = ProgressTracker()
        received: list[float] = []

        async def consume() -> None:
            async for item in tracker.subscribe(1):
                received.append(item.progress)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)

        for value in (0.25, 0.5, 0.75):
            await tracker.publish(update(progress=value))
        await tracker.publish(terminal())

        await asyncio.wait_for(task, timeout=2.0)
        assert received == [0.25, 0.5, 0.75, 1.0]

    async def test_late_subscriber_gets_current_snapshot(self) -> None:
        """A client attaching mid-run sees state immediately, not on next change."""
        tracker = ProgressTracker()
        await tracker.publish(update(progress=0.6))

        received: list[float] = []

        async def consume() -> None:
            async for item in tracker.subscribe(1):
                received.append(item.progress)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await tracker.publish(terminal())
        await asyncio.wait_for(task, timeout=2.0)

        assert received[0] == 0.6

    async def test_subscribing_after_completion_terminates_immediately(self) -> None:
        tracker = ProgressTracker()
        await tracker.publish(terminal(count=7))

        received = [item async for item in tracker.subscribe(1)]

        assert len(received) == 1
        assert received[0].detection_count == 7

    async def test_multiple_subscribers_each_receive_all_updates(self) -> None:
        tracker = ProgressTracker()

        async def consume() -> list[float]:
            return [item.progress async for item in tracker.subscribe(1)]

        tasks = [asyncio.create_task(consume()) for _ in range(3)]
        await asyncio.sleep(0)

        await tracker.publish(update(progress=0.5))
        await tracker.publish(terminal())

        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        assert all(result == [0.5, 1.0] for result in results)

    async def test_jobs_are_isolated(self) -> None:
        tracker = ProgressTracker()

        async def consume(job_id: int) -> list[int]:
            return [item.job_id async for item in tracker.subscribe(job_id)]

        first = asyncio.create_task(consume(1))
        second = asyncio.create_task(consume(2))
        await asyncio.sleep(0)

        await tracker.publish(terminal(job_id=1))
        await tracker.publish(terminal(job_id=2))

        assert await asyncio.wait_for(first, timeout=2.0) == [1]
        assert await asyncio.wait_for(second, timeout=2.0) == [2]

    async def test_slow_consumer_does_not_block_publisher(self) -> None:
        """Publishing stays responsive when a listener never drains its queue."""
        tracker = ProgressTracker()

        async def stall() -> None:
            async for _ in tracker.subscribe(1):
                await asyncio.sleep(10)

        task = asyncio.create_task(stall())
        await asyncio.sleep(0)

        # Far more updates than the queue bound; must not hang.
        for index in range(500):
            await asyncio.wait_for(
                tracker.publish(update(progress=index / 500)), timeout=1.0
            )

        task.cancel()

    async def test_release_discards_state_without_listeners(self) -> None:
        tracker = ProgressTracker()
        await tracker.publish(terminal())
        assert await tracker.latest(1) is not None

        await tracker.release(1)
        assert await tracker.latest(1) is None
