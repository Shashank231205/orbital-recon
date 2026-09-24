"""Job status, progress streaming and detection results."""

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from orbital_recon.api.deps import SessionDep, TrackerDep
from orbital_recon.api.routes.common import require
from orbital_recon.db.models import Detection, Job
from orbital_recon.ml.postprocess.geometry import OrientedBox
from orbital_recon.schemas.api import (
    ClassCount,
    DetectionListResponse,
    JobOut,
    JobSummaryResponse,
)
from orbital_recon.schemas.detection import BoundingBoxOut, DetectionOut, GeoPoint
from orbital_recon.schemas.enums import ThreatLevel
from orbital_recon.services.progress import ProgressTracker

router = APIRouter(prefix="/jobs", tags=["jobs"])


def to_detection_out(detection: Detection) -> DetectionOut:
    """Serialise a stored detection, including its box corners for drawing."""
    box = OrientedBox(
        cx=detection.cx,
        cy=detection.cy,
        width=detection.width,
        height=detection.height,
        angle=detection.angle,
    )
    corners = [(float(x), float(y)) for x, y in box.corners()]

    centroid = (
        GeoPoint(latitude=detection.latitude, longitude=detection.longitude)
        if detection.latitude is not None and detection.longitude is not None
        else None
    )

    footprint = None
    if detection.footprint_geojson is not None:
        ring = json.loads(detection.footprint_geojson)["coordinates"][0]
        # The stored ring repeats its first position to close the polygon; the
        # client draws from the four distinct corners.
        footprint = [
            GeoPoint(latitude=latitude, longitude=longitude)
            for longitude, latitude in ring[:-1]
        ]

    return DetectionOut(
        id=detection.id,
        class_name=detection.display_name,
        confidence=detection.confidence,
        threat_level=ThreatLevel(detection.threat_level),
        box=BoundingBoxOut(
            cx=detection.cx,
            cy=detection.cy,
            width=detection.width,
            height=detection.height,
            angle=detection.angle,
            corners=corners,
        ),
        centroid=centroid,
        footprint=footprint,
    )


@router.get("/{job_id}", response_model=JobOut)
async def get_job(job_id: int, session: SessionDep) -> Job:
    """Return a job's current state."""
    return await require(session, Job, job_id, "Job")


async def _event_stream(tracker: ProgressTracker, job_id: int) -> AsyncIterator[str]:
    """Render progress updates as server-sent events."""
    async for update in tracker.subscribe(job_id):
        payload = {
            "job_id": update.job_id,
            "status": update.status,
            "stage": update.stage,
            "progress": round(update.progress, 4),
            "message": update.message,
            "detection_count": update.detection_count,
            "error": update.error,
        }
        yield f"data: {json.dumps(payload)}\n\n"


@router.get("/{job_id}/stream")
async def stream_job(job_id: int, session: SessionDep, tracker: TrackerDep) -> StreamingResponse:
    """Stream a job's progress until it reaches a terminal state.

    Server-sent events are used rather than websockets because the traffic is
    one-way and SSE reconnects without extra client code.
    """
    await require(session, Job, job_id, "Job")

    return StreamingResponse(
        _event_stream(tracker, job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Prevents proxies from buffering the stream into silence.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/detections", response_model=DetectionListResponse)
async def list_detections(
    job_id: int,
    session: SessionDep,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    threat_level: Annotated[ThreatLevel | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DetectionListResponse:
    """Return a job's detections, filtered and ordered by confidence."""
    await require(session, Job, job_id, "Job")

    conditions = [Detection.job_id == job_id, Detection.confidence >= min_confidence]
    if threat_level is not None:
        conditions.append(Detection.threat_level == threat_level)

    total = await session.scalar(
        select(func.count(Detection.id)).where(*conditions)
    )
    rows = await session.scalars(
        select(Detection)
        .where(*conditions)
        .order_by(Detection.confidence.desc())
        .limit(limit)
        .offset(offset)
    )

    return DetectionListResponse(
        job_id=job_id,
        total=total or 0,
        items=[to_detection_out(row) for row in rows],
    )


@router.get("/{job_id}/summary", response_model=JobSummaryResponse)
async def summarise_job(job_id: int, session: SessionDep) -> JobSummaryResponse:
    """Aggregate a job's detections by class and threat level."""
    job = await require(session, Job, job_id, "Job")

    rows = (
        await session.execute(
            select(
                Detection.class_label,
                Detection.display_name,
                Detection.threat_level,
                func.count(Detection.id),
            )
            .where(Detection.job_id == job_id)
            .group_by(Detection.class_label, Detection.display_name, Detection.threat_level)
            .order_by(func.count(Detection.id).desc())
        )
    ).all()

    by_class = [
        ClassCount(
            label=label,
            display_name=display_name,
            threat_level=ThreatLevel(threat_level),
            count=count,
        )
        for label, display_name, threat_level, count in rows
    ]

    by_threat: dict[str, int] = {}
    for entry in by_class:
        by_threat[entry.threat_level] = by_threat.get(entry.threat_level, 0) + entry.count

    mean_confidence = await session.scalar(
        select(func.avg(Detection.confidence)).where(Detection.job_id == job_id)
    )

    return JobSummaryResponse(
        job_id=job_id,
        scene_id=job.scene_id,
        total_detections=sum(entry.count for entry in by_class),
        mean_confidence=float(mean_confidence or 0.0),
        by_class=by_class,
        by_threat=by_threat,
    )
