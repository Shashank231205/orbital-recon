"""Shared request dependencies.

Long-lived collaborators, the LLM router and the analysis pipeline, are created
once during startup and stored on the application state. Requests read them from
there rather than constructing their own, which keeps a single detector in
memory and a single set of provider health counters.
"""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from orbital_recon.core.config import Settings, get_settings
from orbital_recon.db.session import get_db
from orbital_recon.llm.router import LLMRouter
from orbital_recon.services.intelligence import IntelligenceService
from orbital_recon.services.pipeline import AnalysisPipeline
from orbital_recon.services.progress import ProgressTracker


def get_router(request: Request) -> LLMRouter:
    return request.app.state.llm_router


def get_pipeline(request: Request) -> AnalysisPipeline:
    return request.app.state.pipeline


def get_tracker(request: Request) -> ProgressTracker:
    return request.app.state.tracker


def get_intelligence(
    router: Annotated[LLMRouter, Depends(get_router)],
) -> IntelligenceService:
    return IntelligenceService(router)


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_db)]
RouterDep = Annotated[LLMRouter, Depends(get_router)]
PipelineDep = Annotated[AnalysisPipeline, Depends(get_pipeline)]
TrackerDep = Annotated[ProgressTracker, Depends(get_tracker)]
IntelligenceDep = Annotated[IntelligenceService, Depends(get_intelligence)]
