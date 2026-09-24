"""Shared enumerations for the detection domain."""

from enum import Enum


class Modality(str, Enum):
    """Sensor modality of a source image."""

    ELECTRO_OPTICAL = "eo"
    INFRARED = "ir"
    SAR = "sar"


class JobStatus(str, Enum):
    """Lifecycle of an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(str, Enum):
    """Pipeline stages reported to the client as a job progresses."""

    QUEUED = "queued"
    LOADING = "loading"
    PREPROCESSING = "preprocessing"
    TILING = "tiling"
    INFERENCE = "inference"
    MERGING = "merging"
    GEOREFERENCING = "georeferencing"
    PERSISTING = "persisting"
    DONE = "done"


class ThreatLevel(str, Enum):
    """Operational priority assigned to a detected asset class."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"
