"""Build a georeferenced sample scene from an ordinary image.

The repository ships a plain JPEG rather than a GeoTIFF, because the GeoTIFF is
an order of magnitude larger and carries no information the transform below does
not already describe. Run this to produce the georeferenced counterpart used to
exercise the map path.

The transform places the scene near Mumbai at 0.5 m per pixel in UTM zone 43N.
The coordinates are internally consistent but synthetic: this is not a real
acquisition, and the imagery was not taken at that location.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

# UTM 43N easting and northing near Mumbai harbour.
DEFAULT_EASTING = 285000.0
DEFAULT_NORTHING = 2110000.0
DEFAULT_RESOLUTION = 0.5
DEFAULT_CRS = "EPSG:32643"


def build(source: Path, destination: Path, resolution: float) -> None:
    """Write ``source`` as a GeoTIFF carrying a synthetic affine transform."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read {source}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=DEFAULT_CRS,
        transform=from_origin(
            west=DEFAULT_EASTING,
            north=DEFAULT_NORTHING,
            xsize=resolution,
            ysize=resolution,
        ),
        compress="deflate",
    ) as dataset:
        dataset.write(np.transpose(rgb, (2, 0, 1)))

    print(f"Wrote {destination} ({width}x{height}, {DEFAULT_CRS}, {resolution} m/px)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("samples/harbour-eo.jpg"),
        help="Image to georeference.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("samples/harbour-georeferenced.tif"),
        help="Output GeoTIFF.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help="Ground sample distance in metres per pixel.",
    )
    arguments = parser.parse_args()

    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    build(arguments.source, arguments.destination, arguments.resolution)


if __name__ == "__main__":
    main()
