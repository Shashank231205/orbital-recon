"""Service health and capability reporting."""

from fastapi import APIRouter

from orbital_recon import __version__
from orbital_recon.api.deps import PipelineDep, RouterDep
from orbital_recon.ml.registry.loader import resolve_device
from orbital_recon.schemas.api import HealthResponse, ProviderStatusOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(pipeline: PipelineDep, llm: RouterDep) -> HealthResponse:
    """Report what the service can currently do.

    Provider health comes from cached state rather than live probes so that a
    frequently polled endpoint does not generate outbound traffic of its own.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        device=resolve_device(pipeline.device_preference),
        detector_loaded=pipeline.is_detector_loaded,
        llm_providers=[
            ProviderStatusOut(
                name=status.name,
                configured=status.configured,
                healthy=status.healthy,
                consecutive_failures=status.consecutive_failures,
            )
            for status in llm.status()
        ],
    )


@router.post("/health/providers/refresh", response_model=list[ProviderStatusOut])
async def refresh_providers(llm: RouterDep) -> list[ProviderStatusOut]:
    """Probe every language model provider and return fresh status."""
    return [
        ProviderStatusOut(
            name=status.name,
            configured=status.configured,
            healthy=status.healthy,
            consecutive_failures=status.consecutive_failures,
        )
        for status in await llm.refresh_health()
    ]
