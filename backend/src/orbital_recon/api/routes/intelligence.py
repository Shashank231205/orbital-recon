"""Natural-language querying over detection results."""

from fastapi import APIRouter
from sqlalchemy import select

from orbital_recon.api.deps import IntelligenceDep, SessionDep
from orbital_recon.api.routes.common import require
from orbital_recon.db.models import Detection, Job, Scene
from orbital_recon.schemas.api import BriefResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/jobs", tags=["intelligence"])


async def _load_context(session: SessionDep, job_id: int) -> tuple[Job, Scene, list[Detection]]:
    """Load a job with its scene and detections."""
    job = await require(session, Job, job_id, "Job")
    scene = await require(session, Scene, job.scene_id, "Scene")
    detections = list(
        await session.scalars(
            select(Detection)
            .where(Detection.job_id == job_id)
            .order_by(Detection.confidence.desc())
        )
    )
    return job, scene, detections


@router.post("/{job_id}/query", response_model=QueryResponse)
async def query_job(
    job_id: int,
    payload: QueryRequest,
    session: SessionDep,
    intelligence: IntelligenceDep,
) -> QueryResponse:
    """Answer a question about a job's findings."""
    job, scene, detections = await _load_context(session, job_id)
    result = await intelligence.answer(payload.question, job, scene, detections)

    return QueryResponse(
        job_id=job_id,
        question=payload.question,
        answer=result.text,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@router.post("/{job_id}/brief", response_model=BriefResponse)
async def brief_job(
    job_id: int,
    session: SessionDep,
    intelligence: IntelligenceDep,
) -> BriefResponse:
    """Generate an intelligence brief for a job."""
    job, scene, detections = await _load_context(session, job_id)
    result = await intelligence.brief(job, scene, detections)

    return BriefResponse(
        job_id=job_id,
        brief=result.text,
        provider=result.provider,
        model=result.model,
        latency_ms=result.latency_ms,
    )
