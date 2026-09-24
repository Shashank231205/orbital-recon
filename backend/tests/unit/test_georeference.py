"""Tests for pixel to geographic coordinate conversion."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from orbital_recon.ml.postprocess.geometry import OrientedBox
from orbital_recon.ml.postprocess.georeference import GeoReferencer, read_geo_context


@pytest.fixture
def geotiff(tmp_path: Path) -> Path:
    """A small UTM 43N scene over Bengaluru at 1 metre resolution."""
    path = tmp_path / "scene.tif"
    transform = from_origin(west=675000.0, north=1440000.0, xsize=1.0, ysize=1.0)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=transform,
    ) as dataset:
        dataset.write(np.zeros((3, 256, 256), dtype=np.uint8))

    return path


@pytest.fixture
def plain_image(tmp_path: Path) -> Path:
    """A raster with no CRS, which rasterio warns about by design."""
    path = tmp_path / "plain.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8"
    ) as dataset:
        dataset.write(np.zeros((1, 64, 64), dtype=np.uint8))
    return path


class TestReadGeoContext:
    def test_reads_georeferenced_scene(self, geotiff: Path) -> None:
        context = read_geo_context(geotiff)
        assert context is not None
        assert context.crs == "EPSG:32643"
        assert (context.width, context.height) == (256, 256)

    def test_returns_none_without_georeferencing(self, plain_image: Path) -> None:
        assert read_geo_context(plain_image) is None

    def test_returns_none_for_unreadable_file(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.tif"
        broken.write_bytes(b"not a raster")
        assert read_geo_context(broken) is None


class TestGeoReferencer:
    def test_origin_maps_to_expected_location(self, geotiff: Path) -> None:
        context = read_geo_context(geotiff)
        assert context is not None
        point = GeoReferencer(context).pixel_to_geo(0.0, 0.0)
        # UTM 43N easting 675000 / northing 1440000 resolves to Karnataka, India.
        assert point.latitude == pytest.approx(13.0209, abs=1e-3)
        assert point.longitude == pytest.approx(76.6137, abs=1e-3)

    def test_moving_east_increases_longitude(self, geotiff: Path) -> None:
        context = read_geo_context(geotiff)
        assert context is not None
        referencer = GeoReferencer(context)
        assert referencer.pixel_to_geo(200.0, 0.0).longitude > referencer.pixel_to_geo(
            0.0, 0.0
        ).longitude

    def test_moving_south_decreases_latitude(self, geotiff: Path) -> None:
        context = read_geo_context(geotiff)
        assert context is not None
        referencer = GeoReferencer(context)
        assert referencer.pixel_to_geo(0.0, 200.0).latitude < referencer.pixel_to_geo(
            0.0, 0.0
        ).latitude

    def test_footprint_has_four_corners(self, geotiff: Path) -> None:
        context = read_geo_context(geotiff)
        assert context is not None
        box = OrientedBox(128.0, 128.0, 40.0, 20.0, 0.5)
        assert len(GeoReferencer(context).box_footprint(box)) == 4

    def test_centroid_lies_inside_footprint_bounds(self, geotiff: Path) -> None:
        context = read_geo_context(geotiff)
        assert context is not None
        referencer = GeoReferencer(context)
        box = OrientedBox(128.0, 128.0, 40.0, 20.0, 0.3)

        centroid = referencer.box_centroid(box)
        footprint = referencer.box_footprint(box)
        latitudes = [p.latitude for p in footprint]
        longitudes = [p.longitude for p in footprint]

        assert min(latitudes) <= centroid.latitude <= max(latitudes)
        assert min(longitudes) <= centroid.longitude <= max(longitudes)

    def test_one_pixel_is_one_metre(self, geotiff: Path) -> None:
        """A 1 m scene should place adjacent pixels roughly 1 m apart."""
        context = read_geo_context(geotiff)
        assert context is not None
        referencer = GeoReferencer(context)

        first = referencer.pixel_to_geo(0.0, 0.0)
        hundred_east = referencer.pixel_to_geo(100.0, 0.0)
        metres_per_degree_lon = 111_320.0 * np.cos(np.radians(first.latitude))
        distance = (hundred_east.longitude - first.longitude) * metres_per_degree_lon

        assert distance == pytest.approx(100.0, rel=0.02)


class TestGeoContext:
    def test_is_immutable(self, geotiff: Path) -> None:
        context = read_geo_context(geotiff)
        assert context is not None
        with pytest.raises(AttributeError):
            context.crs = "EPSG:4326"  # type: ignore[misc]
