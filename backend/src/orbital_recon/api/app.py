"""Application factory and lifecycle."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from orbital_recon import __version__
from orbital_recon.api.routes import health, intelligence, jobs, scenes
from orbital_recon.core.config import Settings, get_settings
from orbital_recon.core.exceptions import OrbitalReconError
from orbital_recon.core.logging import configure_logging, get_logger
from orbital_recon.db.session import create_schema, dispose_engine, init_engine
from orbital_recon.llm.router import LLMRouter
from orbital_recon.services.pipeline import AnalysisPipeline
from orbital_recon.services.progress import ProgressTracker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare shared resources on startup and release them on shutdown.

    Model weights are deliberately not loaded here: startup stays fast, and a
    deployment that only serves queries never pays for a detector it will not
    use.
    """
    settings: Settings = app.state.settings
    settings.ensure_directories()

    engine = init_engine(settings)
    await create_schema(engine)

    app.state.tracker = ProgressTracker()
    app.state.llm_router = LLMRouter(settings)
    app.state.pipeline = AnalysisPipeline(settings, app.state.tracker)

    logger.info(
        "service_started",
        version=__version__,
        environment=settings.environment,
        llm_configured=app.state.llm_router.has_configured_provider,
    )

    try:
        yield
    finally:
        await app.state.llm_router.aclose()
        await dispose_engine()
        logger.info("service_stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Overrides the environment-derived configuration. Used by tests
            to point at temporary directories and an isolated database.
    """
    resolved = settings or get_settings()
    configure_logging(resolved)

    app = FastAPI(
        title="Orbital Recon",
        description=(
            "Multi-modal satellite target detection and geospatial intelligence."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(OrbitalReconError)
    async def handle_domain_error(
        request: Request, exc: OrbitalReconError
    ) -> JSONResponse:
        """Translate domain errors into their declared status codes."""
        logger.warning(
            "request_failed",
            path=request.url.path,
            code=exc.code,
            message=exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "details": exc.details},
        )

    for module in (health, scenes, jobs, intelligence):
        app.include_router(module.router, prefix=resolved.api_prefix)

    return app


app = create_app()
