"""Conversion of pixel detections into geographic coordinates.

A detection is only actionable once it carries a real-world position. GeoTIFF
scenes embed an affine transform mapping pixel indices to projected coordinates
plus the coordinate reference system those coordinates belong to. Projected
coordinates are then reprojected to WGS84, because mapping clients and
downstream reporting both expect latitude and longitude.

Scenes without embedded geospatial metadata, such as an ordinary JPEG, are
handled by returning ``None`` rather than fabricating a position.
"""

from dataclasses import dataclass
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.transform import Affine, xy

from orbital_recon.core.logging import get_logger
from orbital_recon.ml.postprocess.geometry import OrientedBox
from orbital_recon.schemas.detection import GeoPoint

logger = get_logger(__name__)

WGS84 = "EPSG:4326"


@dataclass(frozen=True, slots=True)
class GeoContext:
    """Geospatial metadata extracted from a source scene."""

    transform: Affine
    crs: str
    width: int
    height: int


class GeoReferencer:
    """Maps pixel coordinates to WGS84 latitude and longitude.

    The projection transformer is built once per scene; constructing one per
    detection would dominate the cost of geo-referencing a dense scene.
    """

    def __init__(self, context: GeoContext) -> None:
        self._transform = context.transform
        self._to_wgs84 = Transformer.from_crs(context.crs, WGS84, always_xy=True)

    def pixel_to_geo(self, x: float, y: float) -> GeoPoint:
        """Convert one pixel coordinate to a WGS84 point.

        Note that ``rasterio.transform.xy`` takes row before column, so the
        arguments are passed as ``(y, x)``.
        """
        easting, northing = xy(self._transform, y, x, offset="center")
        longitude, latitude = self._to_wgs84.transform(easting, northing)
        return GeoPoint(latitude=latitude, longitude=longitude)

    def box_centroid(self, box: OrientedBox) -> GeoPoint:
        """Geographic position of a detection's centre."""
        return self.pixel_to_geo(box.cx, box.cy)

    def box_footprint(self, box: OrientedBox) -> list[GeoPoint]:
        """Geographic outline of a detection, corner by corner."""
        return [self.pixel_to_geo(float(x), float(y)) for x, y in box.corners()]


def read_geo_context(path: Path) -> GeoContext | None:
    """Extract geospatial metadata from a scene.

    Returns:
        The scene's ``GeoContext``, or ``None`` when the file carries no usable
        georeferencing, in which case detections stay in pixel space.
    """
    try:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                logger.info("scene_not_georeferenced", path=path.name, reason="missing_crs")
                return None

            # rasterio synthesises an identity transform for plain images; it
            # carries no geographic meaning and must not be trusted.
            if dataset.transform == Affine.identity():
                logger.info("scene_not_georeferenced", path=path.name, reason="identity_transform")
                return None

            return GeoContext(
                transform=dataset.transform,
                crs=dataset.crs.to_string(),
                width=dataset.width,
                height=dataset.height,
            )
    except rasterio.errors.RasterioError as exc:
        logger.warning("geo_context_read_failed", path=path.name, error=str(exc))
        return None
