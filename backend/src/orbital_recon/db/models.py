"""Database models.

A scene is an uploaded image. A job tracks one analysis run over a scene and
carries the progress a client polls or streams. Detections belong to a job
rather than directly to a scene so that re-running analysis with different
thresholds produces a comparable, independently addressable result set.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from orbital_recon.schemas.enums import JobStage, JobStatus, Modality, ThreatLevel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Scene(Base):
    """An uploaded source image and its geospatial metadata."""

    __tablename__ = "scenes"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(1024))
    modality: Mapped[Modality] = mapped_column(String(16))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    crs: Mapped[str | None] = mapped_column(String(64), default=None)
    is_georeferenced: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list["Job"]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )


class Job(Base):
    """One analysis run over a scene."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"))
    status: Mapped[JobStatus] = mapped_column(String(16), default=JobStatus.PENDING)
    stage: Mapped[JobStage] = mapped_column(String(32), default=JobStage.QUEUED)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(String(512), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    confidence_threshold: Mapped[float] = mapped_column(Float)
    detection_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    scene: Mapped[Scene] = relationship(back_populates="jobs")
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_jobs_status_created", "status", "created_at"),)


class Detection(Base):
    """A single detected asset, in both pixel and geographic coordinates."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))

    class_label: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(64))
    threat_level: Mapped[ThreatLevel] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)

    # Oriented box in scene pixel coordinates.
    cx: Mapped[float] = mapped_column(Float)
    cy: Mapped[float] = mapped_column(Float)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)
    angle: Mapped[float] = mapped_column(Float)

    # WGS84 position, absent when the source scene carries no georeferencing.
    latitude: Mapped[float | None] = mapped_column(Float, default=None)
    longitude: Mapped[float | None] = mapped_column(Float, default=None)
    footprint_geojson: Mapped[str | None] = mapped_column(Text, default=None)

    job: Mapped[Job] = relationship(back_populates="detections")

    __table_args__ = (
        Index("ix_detections_job_confidence", "job_id", "confidence"),
        Index("ix_detections_job_class", "job_id", "class_label"),
    )
