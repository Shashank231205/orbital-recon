"""Scene upload and analysis endpoints."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status

from orbital_recon.api.deps import PipelineDep, SessionDep, SettingsDep
from orbital_recon.api.routes.common import require
from orbital_recon.core.logging import get_logger
from orbital_recon.db.models import Job, Scene
from orbital_recon.schemas.api import SceneOut, UploadResponse
from orbital_recon.schemas.enums import Modality
from orbital_recon.services.scenes import build_scene, store_upload, validate_upload

logger = get_logger(__name__)

router = APIRouter(prefix="/scenes", tags=["scenes"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_scene(
    session: SessionDep,
    settings: SettingsDep,
    pipeline: PipelineDep,
    file: Annotated[UploadFile, File(description="Satellite or drone image")],
    modality: Annotated[Modality, Form()] = Modality.ELECTRO_OPTICAL,
    confidence: Annotated[float | None, Form(ge=0.0, le=1.0)] = None,
) -> UploadResponse:
    """Accept a scene, queue its analysis, and return the job to follow.

    Returns 202 rather than 201 because analysis continues after the response;
    the client watches the job stream for completion.
    """
    content = await file.read()
    suffix = validate_upload(file.filename or "", len(content), settings.max_upload_bytes)
    stored_path = store_upload(content, suffix, settings.upload_dir)

    scene = build_scene(file.filename or stored_path.name, stored_path, modality, len(content))
    session.add(scene)
    await session.flush()

    job = Job(
        scene_id=scene.id,
        confidence_threshold=confidence
        if confidence is not None
        else settings.confidence_threshold,
    )
    session.add(job)
    await session.commit()

    # Detached from the request: the client follows progress over the job stream.
    asyncio.create_task(pipeline.run(job.id))  # noqa: RUF006

    logger.info("analysis_queued", job_id=job.id, scene_id=scene.id)
    return UploadResponse.model_validate({"scene": scene, "job": job}, from_attributes=True)


@router.get("/{scene_id}", response_model=SceneOut)
async def get_scene(scene_id: int, session: SessionDep) -> Scene:
    """Return one scene's metadata."""
    return await require(session, Scene, scene_id, "Scene")
