"""Scene analysis orchestration.

Runs the full path from a stored image to persisted, geo-referenced detections,
publishing progress at each stage so a client can follow the run.

Inference is CPU- and GPU-bound and would block the event loop, so it executes
in a worker thread. Progress from that thread is marshalled back onto the loop
rather than published directly, because the tracker's state is guarded by an
asyncio lock that is not thread-safe.
"""

import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np

from orbital_recon.core.config import Settings
from orbital_recon.core.exceptions import OrbitalReconError, ValidationError
from orbital_recon.core.logging import get_logger
from orbital_recon.db.models import Detection, Job, Scene
from orbital_recon.db.session import session_scope
from orbital_recon.ml.detectors.obb_detector import ObbDetector
from orbital_recon.ml.detectors.taxonomy import resolve_asset_class
from orbital_recon.ml.postprocess.georeference import GeoReferencer, read_geo_context
from orbital_recon.ml.postprocess.nms import merge_tile_detections
from orbital_recon.ml.preprocessing.enhancement import preprocess
from orbital_recon.schemas.detection import GeoPoint, RawDetection
from orbital_recon.schemas.enums import JobStage, JobStatus, Modality
from orbital_recon.services.progress import ProgressTracker, ProgressUpdate

logger = get_logger(__name__)

# Fraction of the progress bar allotted to inference, which dominates runtime.
_INFERENCE_SPAN = (0.25, 0.85)


def load_image(path: Path) -> np.ndarray:
    """Read a scene from disk as an RGB array.

    Raises:
        ValidationError: If the file cannot be decoded as an image.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValidationError(f"Could not decode image: {path.name}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class AnalysisPipeline:
    """Executes analysis jobs and records their results."""

    def __init__(self, settings: Settings, tracker: ProgressTracker) -> None:
        self._settings = settings
        self._tracker = tracker
        self._detector: ObbDetector | None = None

    @property
    def device_preference(self) -> str:
        """Configured device preference, before resolution."""
        return self._settings.detection_device

    @property
    def is_detector_loaded(self) -> bool:
        """Whether weights are resident, since loading is deferred to first use."""
        return self._detector is not None

    def _get_detector(self) -> ObbDetector:
        """Build the detector on first use so startup stays fast."""
        if self._detector is None:
            self._detector = ObbDetector(
                weights=self._settings.detector_weights,
                device=self._settings.detection_device,
                confidence_threshold=self._settings.confidence_threshold,
                tile_size=self._settings.tile_size,
                tile_overlap=self._settings.tile_overlap,
            )
        return self._detector

    async def _publish(
        self,
        job_id: int,
        stage: JobStage,
        progress: float,
        message: str | None = None,
        *,
        status: JobStatus = JobStatus.RUNNING,
        detection_count: int = 0,
        error: str | None = None,
    ) -> None:
        """Record a stage transition and notify listeners.

        The database write happens before the notification so that a client
        reacting to a terminal update always reads a job row that agrees with
        it. Notifying first leaves a window where the stream says completed but
        the stored job still says running.
        """
        async with session_scope() as session:
            job = await session.get(Job, job_id)
            if job is not None:
                job.status = status
                job.stage = stage
                job.progress = progress
                job.message = message
                if error is not None:
                    job.error = error
                if detection_count:
                    job.detection_count = detection_count

        await self._tracker.publish(
            ProgressUpdate(
                job_id=job_id,
                status=status,
                stage=stage,
                progress=progress,
                message=message,
                detection_count=detection_count,
                error=error,
            )
        )

    def _run_detection(
        self,
        image: np.ndarray,
        job_id: int,
        loop: asyncio.AbstractEventLoop,
    ) -> list[tuple[RawDetection, int, int]]:
        """Run tiled inference, reporting tile progress back to the event loop."""
        low, high = _INFERENCE_SPAN

        def on_progress(done: int, total: int) -> None:
            fraction = low + (high - low) * (done / max(total, 1))
            asyncio.run_coroutine_threadsafe(
                self._publish(
                    job_id,
                    JobStage.INFERENCE,
                    fraction,
                    f"Analysed {done} of {total} tiles",
                ),
                loop,
            )

        return self._get_detector().detect(image, on_progress=on_progress)

    def _to_records(
        self,
        detections: list[RawDetection],
        job_id: int,
        referencer: GeoReferencer | None,
    ) -> list[Detection]:
        """Convert detections into database rows, attaching geography if available."""
        records: list[Detection] = []

        for detection in detections:
            asset = resolve_asset_class(detection.class_name)
            record = Detection(
                job_id=job_id,
                class_label=asset.label,
                display_name=asset.display_name,
                threat_level=asset.threat_level,
                confidence=detection.confidence,
                cx=detection.box.cx,
                cy=detection.box.cy,
                width=detection.box.width,
                height=detection.box.height,
                angle=detection.box.angle,
            )

            if referencer is not None:
                centroid = referencer.box_centroid(detection.box)
                record.latitude = centroid.latitude
                record.longitude = centroid.longitude
                record.footprint_geojson = json.dumps(
                    _footprint_to_geojson(referencer.box_footprint(detection.box))
                )

            records.append(record)

        return records

    async def run(self, job_id: int) -> None:
        """Execute one analysis job.

        Failures are recorded on the job and published to listeners; they are not
        re-raised, because this runs detached from any request.
        """
        started = time.perf_counter()

        try:
            async with session_scope() as session:
                job = await session.get(Job, job_id)
                if job is None:
                    raise ValidationError(f"Job {job_id} does not exist")
                scene = await session.get(Scene, job.scene_id)
                if scene is None:
                    raise ValidationError(f"Scene {job.scene_id} does not exist")
                scene_path = Path(scene.stored_path)
                modality = Modality(scene.modality)

            await self._publish(job_id, JobStage.LOADING, 0.05, "Loading scene")
            image = await asyncio.to_thread(load_image, scene_path)

            await self._publish(job_id, JobStage.PREPROCESSING, 0.15, "Enhancing imagery")
            prepared = await asyncio.to_thread(preprocess, image, modality)

            await self._publish(job_id, JobStage.TILING, 0.25, "Tiling scene")
            loop = asyncio.get_running_loop()
            raw = await asyncio.to_thread(self._run_detection, prepared, job_id, loop)

            await self._publish(job_id, JobStage.MERGING, 0.88, "Merging overlapping tiles")
            merged = await asyncio.to_thread(
                merge_tile_detections, raw, self._settings.iou_threshold
            )

            await self._publish(job_id, JobStage.GEOREFERENCING, 0.93, "Locating targets")
            context = await asyncio.to_thread(read_geo_context, scene_path)
            referencer = GeoReferencer(context) if context is not None else None

            await self._publish(job_id, JobStage.PERSISTING, 0.97, "Saving detections")
            records = self._to_records(merged, job_id, referencer)
            duration_ms = int((time.perf_counter() - started) * 1000)

            async with session_scope() as session:
                session.add_all(records)
                stored = await session.get(Job, job_id)
                if stored is not None:
                    stored.detection_count = len(records)
                    stored.duration_ms = duration_ms

            await self._publish(
                job_id,
                JobStage.DONE,
                1.0,
                f"Detected {len(records)} assets",
                status=JobStatus.COMPLETED,
                detection_count=len(records),
            )
            logger.info(
                "job_completed",
                job_id=job_id,
                detections=len(records),
                duration_ms=duration_ms,
            )

        except OrbitalReconError as exc:
            logger.warning("job_failed", job_id=job_id, error=exc.message)
            await self._publish(
                job_id,
                JobStage.DONE,
                1.0,
                "Analysis failed",
                status=JobStatus.FAILED,
                error=exc.message,
            )
        except Exception as exc:
            logger.exception("job_failed_unexpectedly", job_id=job_id)
            await self._publish(
                job_id,
                JobStage.DONE,
                1.0,
                "Analysis failed",
                status=JobStatus.FAILED,
                error=f"Unexpected error: {exc}",
            )


def _footprint_to_geojson(footprint: list[GeoPoint]) -> dict[str, object]:
    """Build a closed GeoJSON polygon from a detection footprint.

    GeoJSON requires the first and last position of a linear ring to be
    identical, so the opening corner is repeated at the end.
    """
    ring = [[point.longitude, point.latitude] for point in footprint]
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}
