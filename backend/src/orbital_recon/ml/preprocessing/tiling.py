"""Sliding-window tiling for large overhead imagery.

Satellite scenes routinely exceed 10000x10000 pixels while detectors expect
inputs near 640x640. Downscaling a whole scene destroys the small objects this
system targets, so scenes are cut into overlapping tiles, inferred on
independently, and merged back in image coordinates. The overlap guarantees that
an object straddling a tile boundary appears whole in at least one neighbour.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Tile:
    """A tile crop plus the offset needed to map detections back to the scene."""

    image: np.ndarray
    x_offset: int
    y_offset: int
    index: int


def _axis_origins(extent: int, window: int, stride: int) -> list[int]:
    """Return window start positions covering ``extent`` with no gaps.

    The final origin is clamped so the last window ends exactly at the edge,
    which avoids emitting a narrow remainder tile that would be mostly padding.
    """
    if extent <= window:
        return [0]

    origins = list(range(0, extent - window + 1, stride))
    last_covered = origins[-1] + window
    if last_covered < extent:
        origins.append(extent - window)
    return origins


def generate_tiles(
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> Iterator[Tile]:
    """Yield overlapping tiles covering the full image.

    Args:
        image: Scene as an ``(H, W, C)`` array.
        tile_size: Square tile edge length in pixels.
        overlap: Fraction of ``tile_size`` shared between adjacent tiles.

    Raises:
        ValueError: If ``tile_size`` is not positive or ``overlap`` is outside [0, 1).
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    height, width = image.shape[:2]
    stride = max(1, int(tile_size * (1.0 - overlap)))

    index = 0
    for y in _axis_origins(height, tile_size, stride):
        for x in _axis_origins(width, tile_size, stride):
            crop = image[y : y + tile_size, x : x + tile_size]
            yield Tile(image=crop, x_offset=x, y_offset=y, index=index)
            index += 1


def count_tiles(height: int, width: int, tile_size: int, overlap: float) -> int:
    """Number of tiles ``generate_tiles`` would emit, for progress reporting."""
    stride = max(1, int(tile_size * (1.0 - overlap)))
    return len(_axis_origins(height, tile_size, stride)) * len(
        _axis_origins(width, tile_size, stride)
    )
