"""Download and derive a set of sample scenes.

Sources are public imagery released for open use. Each entry names the modality
it should be analysed as, so a reviewer can exercise every preprocessing path
without hunting for suitable data.

Infrared and radar scenes are derived rather than downloaded: genuine IR and SAR
products are large and awkwardly licensed, while the preprocessing path under
test is the same either way. Derivation is labelled as such in the manifest so
nobody mistakes a simulated scene for a real acquisition.
"""

import argparse
import json
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import truststore

    _SSL_CONTEXT: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
except ImportError:  # pragma: no cover - environment dependent
    _SSL_CONTEXT = ssl.create_default_context()


@dataclass(frozen=True)
class Sample:
    """One scene to produce."""

    name: str
    modality: str
    description: str
    expect: str
    url: str | None = None
    derive_from: str | None = None
    transform: str | None = None
    georeference: bool = False
    tags: list[str] = field(default_factory=list)


SAMPLES: tuple[Sample, ...] = (
    Sample(
        name="01-harbour-vessels.jpg",
        modality="eo",
        url="https://www.ultralytics.com/images/boats.jpg",
        description="Dense marina, vessels moored at close spacing.",
        expect="~210 detections, overwhelmingly vessels, with a few harbour structures.",
        tags=["dense", "vessels", "tile-seams"],
    ),
    Sample(
        name="02-harbour-georeferenced.tif",
        modality="eo",
        derive_from="01-harbour-vessels.jpg",
        transform="georeference",
        georeference=True,
        description="The same marina carrying a coordinate reference.",
        expect="Identical detections, now on the map with WGS84 coordinates.",
        tags=["geotiff", "map", "localisation"],
    ),
    Sample(
        name="03-harbour-infrared.png",
        modality="ir",
        derive_from="01-harbour-vessels.jpg",
        transform="infrared",
        description="Thermal simulation: single channel with a compressed range.",
        expect=(
            "Around 45 detections, mostly reclassified as vehicles. Removing "
            "colour costs the model the cues that separate vessels from ground "
            "assets, so the class mix shifts as well as the count."
        ),
        tags=["infrared", "clahe", "percentile-stretch"],
    ),
    Sample(
        name="04-harbour-radar.png",
        modality="sar",
        derive_from="01-harbour-vessels.jpg",
        transform="radar",
        description="Radar simulation: multiplicative speckle over a grey scene.",
        expect=(
            "Only a handful of detections. The speckle here is heavier than a "
            "real radar product, which shows the limit of the approach: the Lee "
            "filter preserves edges but cannot recover texture already lost."
        ),
        tags=["sar", "speckle", "lee-filter"],
    ),
    Sample(
        name="05-harbour-strip.png",
        modality="eo",
        derive_from="01-harbour-vessels.jpg",
        transform="strip",
        description="A wide, shallow crop with an extreme aspect ratio.",
        expect="A single row of tiles; exercises non-square tiling geometry.",
        tags=["aspect-ratio", "tiling"],
    ),
    Sample(
        name="06-harbour-low-contrast.png",
        modality="eo",
        derive_from="01-harbour-vessels.jpg",
        transform="low_contrast",
        description="Haze simulation: the dynamic range is compressed toward mid grey.",
        expect="Detection survives because CLAHE restores local contrast first.",
        tags=["degraded", "haze", "preprocessing"],
    ),
    Sample(
        name="07-harbour-rotated.png",
        modality="eo",
        derive_from="01-harbour-vessels.jpg",
        transform="rotate",
        description="The marina rotated 37 degrees.",
        expect="Similar count with box angles shifted, showing orientation is tracked.",
        tags=["rotation", "oriented-boxes"],
    ),
    Sample(
        name="08-harbour-quarter.png",
        modality="eo",
        derive_from="01-harbour-vessels.jpg",
        transform="crop",
        description="A single quadrant, smaller than the tiling window.",
        expect="One tile only; exercises the path where no tiling is needed.",
        tags=["small-scene", "single-tile"],
    ),
    Sample(
        name="09-harbour-upscaled.png",
        modality="eo",
        derive_from="01-harbour-vessels.jpg",
        transform="upscale",
        description="Enlarged to roughly four megapixels.",
        expect="Many tiles, visible progress steps, and more tile-seam merges.",
        tags=["large-scene", "many-tiles", "progress"],
    ),
    Sample(
        name="10-open-water.png",
        modality="eo",
        derive_from="01-harbour-vessels.jpg",
        transform="empty_water",
        description="Water only, with all structure removed.",
        expect="Few or no detections; confirms the model is not inventing targets.",
        tags=["negative-control", "false-positives"],
    ),
)


class _PermanentRedirect(urllib.request.HTTPRedirectHandler):
    """Follows 308, which the default handler leaves to the caller.

    urllib handles 301 and 302 but not 308, so a host that moved a path
    permanently while preserving the method raises rather than redirects.
    """

    def http_error_308(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
    ) -> object:
        return self.http_error_301(req, fp, 301, msg, headers)  # type: ignore[arg-type]


def download(url: str, destination: Path) -> bool:
    """Fetch a URL, returning whether it succeeded."""
    opener = urllib.request.build_opener(
        _PermanentRedirect,
        urllib.request.HTTPSHandler(context=_SSL_CONTEXT),
    )
    request = urllib.request.Request(url, headers={"User-Agent": "orbital-recon/0.1"})
    try:
        with opener.open(request, timeout=60) as response:
            destination.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  could not fetch {url}: {exc}")
        return False
    return True


def _to_infrared(image: np.ndarray) -> np.ndarray:
    """Approximate a thermal product: single band, compressed range."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Water reads cold and structures warm, so invert to mimic a white-hot palette.
    thermal = 255 - grey
    compressed = (thermal.astype(np.float32) * 0.55 + 70.0).clip(0, 255)
    return compressed.astype(np.uint8)


def _to_radar(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Approximate a radar product: grey scene carrying multiplicative speckle.

    Speckle is multiplicative rather than additive, which is exactly why an
    averaging filter damages a radar image and the Lee filter does not.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    speckle = rng.gamma(shape=4.0, scale=0.25, size=grey.shape).astype(np.float32)
    return (grey * speckle).clip(0, 255).astype(np.uint8)


def _low_contrast(image: np.ndarray) -> np.ndarray:
    """Compress the range toward mid grey, as atmospheric haze does."""
    return (image.astype(np.float32) * 0.35 + 110.0).clip(0, 255).astype(np.uint8)


def _rotate(image: np.ndarray, degrees: float = 37.0) -> np.ndarray:
    """Rotate about the centre, expanding the canvas so nothing is clipped."""
    height, width = image.shape[:2]
    centre = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    bound_w = int(height * sin + width * cos)
    bound_h = int(height * cos + width * sin)
    matrix[0, 2] += bound_w / 2 - centre[0]
    matrix[1, 2] += bound_h / 2 - centre[1]

    return cv2.warpAffine(image, matrix, (bound_w, bound_h), borderValue=(18, 26, 32))


def _empty_water(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace the scene with plausible open water carrying no targets."""
    height, width = image.shape[:2]
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :] = (72.0, 58.0, 38.0)

    texture = rng.normal(0.0, 7.0, (height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=3.0)
    for channel in range(3):
        base[:, :, channel] += texture

    return base.clip(0, 255).astype(np.uint8)


def _write_geotiff(image: np.ndarray, destination: Path) -> None:
    """Write with a synthetic transform near Mumbai at 0.5 m per pixel."""
    import rasterio
    from rasterio.transform import from_origin

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
        crs="EPSG:32643",
        transform=from_origin(west=285000.0, north=2110000.0, xsize=0.5, ysize=0.5),
        compress="deflate",
    ) as dataset:
        dataset.write(np.transpose(rgb, (2, 0, 1)))


def derive(sample: Sample, source: np.ndarray, destination: Path) -> None:
    """Produce a derived scene from an already downloaded one."""
    rng = np.random.default_rng(1234)

    if sample.transform == "georeference":
        _write_geotiff(source, destination)
        return

    transforms = {
        "infrared": lambda: _to_infrared(source),
        "radar": lambda: _to_radar(source, rng),
        "low_contrast": lambda: _low_contrast(source),
        "rotate": lambda: _rotate(source),
        "crop": lambda: source[: source.shape[0] // 2, : source.shape[1] // 2],
        "upscale": lambda: cv2.resize(source, None, fx=2.1, fy=2.1, interpolation=cv2.INTER_CUBIC),
        "empty_water": lambda: _empty_water(source, rng),
        "strip": lambda: source[source.shape[0] // 3 : source.shape[0] * 2 // 3, :],
    }

    operation = transforms.get(sample.transform or "")
    if operation is None:
        raise ValueError(f"Unknown transform: {sample.transform}")

    cv2.imwrite(str(destination), operation())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples"))
    parser.add_argument(
        "--force", action="store_true", help="Rebuild files that already exist."
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    produced: list[Sample] = []

    for sample in SAMPLES:
        destination = arguments.output / sample.name

        if destination.exists() and not arguments.force:
            print(f"{sample.name}: already present")
            produced.append(sample)
            continue

        print(f"{sample.name}: building")

        if sample.url is not None:
            if not download(sample.url, destination):
                continue
        elif sample.derive_from is not None:
            source_path = arguments.output / sample.derive_from
            source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if source is None:
                print(f"  source {sample.derive_from} is missing; skipped")
                continue
            derive(sample, source, destination)

        produced.append(sample)

    manifest = arguments.output / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "file": sample.name,
                    "modality": sample.modality,
                    "description": sample.description,
                    "expect": sample.expect,
                    "derived": sample.url is None,
                    "georeferenced": sample.georeference,
                    "tags": sample.tags,
                }
                for sample in produced
            ],
            indent=2,
        )
        + "\n"
    )

    print(f"\n{len(produced)} scenes in {arguments.output}")
    print(f"Manifest written to {manifest}")


if __name__ == "__main__":
    main()
