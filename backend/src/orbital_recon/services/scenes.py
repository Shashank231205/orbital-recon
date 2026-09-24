"""Scene ingestion.

Uploaded bytes are validated, written to disk under a generated name, and
recorded with whatever geospatial metadata the file carries. Validation happens
before persistence so a rejected upload leaves nothing behind.
"""

import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from orbital_recon.core.exceptions import (
    PayloadTooLargeError,
    UnsupportedMediaError,
    ValidationError,
)
from orbital_recon.core.logging import get_logger
from orbital_recon.db.models import Scene
from orbital_recon.ml.postprocess.georeference import read_geo_context
from orbital_recon.schemas.enums import Modality

logger = get_logger(__name__)

# Formats OpenCV and rasterio can both decode. GeoTIFF is the only one of these
# that carries georeferencing; the rest yield detections in pixel space.
SUPPORTED_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})


def validate_upload(filename: str, size_bytes: int, max_bytes: int) -> str:
    """Check an upload and return its normalised suffix.

    Raises:
        ValidationError: If the filename is empty.
        UnsupportedMediaError: If the extension is not a supported image format.
        PayloadTooLargeError: If the payload exceeds the configured limit.
    """
    if not filename.strip():
        raise ValidationError("Upload is missing a filename")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedMediaError(
            f"Unsupported image format: {suffix or 'none'}",
            details={"supported": sorted(SUPPORTED_SUFFIXES)},
        )

    if size_bytes > max_bytes:
        raise PayloadTooLargeError(
            f"Upload is {size_bytes} bytes, limit is {max_bytes}",
            details={"size_bytes": size_bytes, "max_bytes": max_bytes},
        )

    if size_bytes == 0:
        raise ValidationError("Upload is empty")

    return suffix


def store_upload(content: bytes, suffix: str, upload_dir: Path) -> Path:
    """Write bytes to a uniquely named file and return its path.

    A generated name avoids collisions between uploads sharing a filename and
    keeps user-supplied text out of the filesystem path.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def read_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of a stored scene.

    Dimensions come from the file header rather than a decode, so measuring a
    large scene costs no more than a small one.

    Raises:
        ValidationError: If the file is not a readable image.
    """
    try:
        with Image.open(path) as image:
            return image.width, image.height
    except (UnidentifiedImageError, OSError) as exc:
        raise ValidationError(f"Could not read image: {path.name}") from exc


def build_scene(
    original_filename: str,
    stored_path: Path,
    modality: Modality,
    size_bytes: int,
) -> Scene:
    """Create a scene record from a stored upload."""
    width, height = read_dimensions(stored_path)
    context = read_geo_context(stored_path)

    scene = Scene(
        filename=original_filename,
        stored_path=str(stored_path),
        modality=modality,
        width=context.width if context is not None else width,
        height=context.height if context is not None else height,
        size_bytes=size_bytes,
        crs=context.crs if context is not None else None,
        is_georeferenced=context is not None,
    )

    logger.info(
        "scene_stored",
        filename=original_filename,
        width=scene.width,
        height=scene.height,
        georeferenced=scene.is_georeferenced,
    )
    return scene
