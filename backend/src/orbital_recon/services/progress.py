"""In-process progress tracking for analysis jobs.

Clients watch a job through a stream rather than polling, so progress updates
must reach them as they happen. Each job gets a broadcast hub that fans updates
out to every listener; the latest snapshot is retained so a client that connects
mid-run receives the current state immediately instead of waiting for the next
change.

State is deliberately in-process: it is cheap, it disappears correctly on
restart, and the durable record of a job already lives in the database.
"""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from orbital_recon.core.logging import get_logger
from orbital_recon.schemas.enums import JobStage, JobStatus

logger = get_logger(__name__)

# Bounds a slow client's backlog. Dropping the oldest update is safe because
# each update carries absolute state rather than a delta.
_QUEUE_MAX = 64


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """A snapshot of a job's state at one moment."""

    job_id: int
    status: JobStatus
    stage: JobStage
    progress: float
    message: str | None = None
    detection_count: int = 0
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED)


@dataclass
class _Hub:
    """Fan-out for a single job."""

    listeners: set[asyncio.Queue[ProgressUpdate]] = field(default_factory=set)
    latest: ProgressUpdate | None = None


class ProgressTracker:
    """Publishes job progress to any number of concurrent listeners."""

    def __init__(self) -> None:
        self._hubs: dict[int, _Hub] = {}
        self._lock = asyncio.Lock()

    async def publish(self, update: ProgressUpdate) -> None:
        """Record an update and deliver it to current listeners."""
        async with self._lock:
            hub = self._hubs.setdefault(update.job_id, _Hub())
            hub.latest = update
            listeners = list(hub.listeners)

        for queue in listeners:
            if queue.full():
                # Drop the stalest update rather than blocking the pipeline.
                queue.get_nowait()
            queue.put_nowait(update)

    async def subscribe(self, job_id: int) -> AsyncIterator[ProgressUpdate]:
        """Yield updates for a job until it reaches a terminal state.

        A snapshot of the current state is emitted first, so a client attaching
        to an in-flight job renders immediately.
        """
        queue: asyncio.Queue[ProgressUpdate] = asyncio.Queue(maxsize=_QUEUE_MAX)

        async with self._lock:
            hub = self._hubs.setdefault(job_id, _Hub())
            hub.listeners.add(queue)
            snapshot = hub.latest

        try:
            if snapshot is not None:
                yield snapshot
                if snapshot.is_terminal:
                    return

            while True:
                update = await queue.get()
                yield update
                if update.is_terminal:
                    return
        finally:
            async with self._lock:
                hub = self._hubs.get(job_id)
                if hub is not None:
                    hub.listeners.discard(queue)

    async def release(self, job_id: int) -> None:
        """Forget a finished job once no listener remains."""
        async with self._lock:
            hub = self._hubs.get(job_id)
            if hub is not None and not hub.listeners:
                del self._hubs[job_id]

    async def latest(self, job_id: int) -> ProgressUpdate | None:
        async with self._lock:
            hub = self._hubs.get(job_id)
            return hub.latest if hub is not None else None


tracker = ProgressTracker()
