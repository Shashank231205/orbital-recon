"""Request and response models for the HTTP API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from orbital_recon.schemas.detection import DetectionOut
from orbital_recon.schemas.enums import JobStage, JobStatus, Modality, ThreatLevel


class SceneOut(BaseModel):
    """An uploaded scene."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    modality: Modality
    width: int
    height: int
    size_bytes: int
    crs: str | None
    is_georeferenced: bool
    created_at: datetime


class JobOut(BaseModel):
    """An analysis job and its current progress."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    scene_id: int
    status: JobStatus
    stage: JobStage
    progress: float = Field(ge=0.0, le=1.0)
    message: str | None
    error: str | None
    detection_count: int
    duration_ms: int | None
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    """Result of uploading a scene and queueing its analysis."""

    scene: SceneOut
    job: JobOut


class DetectionListResponse(BaseModel):
    """A page of detections for one job."""

    job_id: int
    total: int
    items: list[DetectionOut]


class ClassCount(BaseModel):
    """Detection count for one asset class."""

    label: str
    display_name: str
    threat_level: ThreatLevel
    count: int


class JobSummaryResponse(BaseModel):
    """Aggregated findings for a completed job."""

    job_id: int
    scene_id: int
    total_detections: int
    mean_confidence: float
    by_class: list[ClassCount]
    by_threat: dict[str, int]


class QueryRequest(BaseModel):
    """An operator question about a job's findings."""

    question: str = Field(min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    """A grounded answer, annotated with the provider that produced it."""

    job_id: int
    question: str
    answer: str
    provider: str
    model: str
    latency_ms: int


class BriefResponse(BaseModel):
    """A generated intelligence brief."""

    job_id: int
    brief: str
    provider: str
    model: str
    latency_ms: int


class ProviderStatusOut(BaseModel):
    """Health of one language model provider."""

    name: str
    configured: bool
    healthy: bool
    consecutive_failures: int


class HealthResponse(BaseModel):
    """Service health, used by the client to show what is available."""

    status: str
    version: str
    device: str
    detector_loaded: bool
    llm_providers: list[ProviderStatusOut]


class ErrorResponse(BaseModel):
    """A failed request."""

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)
