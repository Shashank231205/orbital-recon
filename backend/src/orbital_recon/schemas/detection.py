"""Detection data structures shared by the ML pipeline and the API."""

from dataclasses import dataclass, replace

from pydantic import BaseModel, ConfigDict, Field

from orbital_recon.ml.postprocess.geometry import OrientedBox
from orbital_recon.schemas.enums import ThreatLevel


@dataclass(frozen=True, slots=True)
class RawDetection:
    """A detection in image pixel coordinates, before geo-referencing.

    Kept as a plain dataclass rather than a Pydantic model because suppression
    constructs these in tight loops where validation overhead is measurable.
    """

    box: OrientedBox
    class_name: str
    confidence: float

    def with_box(self, box: OrientedBox) -> "RawDetection":
        return replace(self, box=box)


class GeoPoint(BaseModel):
    """WGS84 coordinate."""

    model_config = ConfigDict(frozen=True)

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


class BoundingBoxOut(BaseModel):
    """Oriented box serialised for the client."""

    model_config = ConfigDict(frozen=True)

    cx: float
    cy: float
    width: float
    height: float
    angle: float
    corners: list[tuple[float, float]] = Field(min_length=4, max_length=4)


class DetectionOut(BaseModel):
    """A single detected asset returned by the API."""

    model_config = ConfigDict(frozen=True)

    id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    threat_level: ThreatLevel
    box: BoundingBoxOut
    centroid: GeoPoint | None = None
    footprint: list[GeoPoint] | None = None
