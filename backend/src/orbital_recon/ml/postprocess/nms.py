"""Non-maximum suppression for oriented boxes.

Overlapping tiles mean the same object is usually detected several times, once
per tile that contains it. Suppression runs per class so that genuinely
different asset types occupying the same footprint, such as a vehicle parked
beside a radar, are not collapsed into one detection.
"""

from collections import defaultdict
from collections.abc import Sequence

from orbital_recon.ml.postprocess.geometry import OrientedBox, rotated_iou
from orbital_recon.schemas.detection import RawDetection


def _suppress_single_class(
    detections: Sequence[RawDetection],
    iou_threshold: float,
) -> list[RawDetection]:
    """Greedy suppression within one class, highest confidence first."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[RawDetection] = []

    for candidate in ordered:
        if all(
            rotated_iou(candidate.box, retained.box) <= iou_threshold for retained in kept
        ):
            kept.append(candidate)

    return kept


def non_max_suppression(
    detections: Sequence[RawDetection],
    iou_threshold: float = 0.45,
) -> list[RawDetection]:
    """Remove duplicate detections, keeping the highest-confidence instance.

    Args:
        detections: Candidates in scene coordinates.
        iou_threshold: Overlap above which the lower-confidence box is dropped.

    Returns:
        Surviving detections ordered by descending confidence.
    """
    if not detections:
        return []

    by_class: dict[str, list[RawDetection]] = defaultdict(list)
    for detection in detections:
        by_class[detection.class_name].append(detection)

    survivors: list[RawDetection] = []
    for class_detections in by_class.values():
        survivors.extend(_suppress_single_class(class_detections, iou_threshold))

    return sorted(survivors, key=lambda d: d.confidence, reverse=True)


def merge_tile_detections(
    tile_detections: Sequence[tuple[RawDetection, int, int]],
    iou_threshold: float = 0.45,
) -> list[RawDetection]:
    """Translate per-tile detections into scene coordinates and suppress duplicates.

    Args:
        tile_detections: Triples of ``(detection, x_offset, y_offset)`` where the
            offsets are the originating tile's position in the scene.
        iou_threshold: Passed through to suppression.
    """
    translated = [
        detection.with_box(detection.box.translated(float(x_offset), float(y_offset)))
        for detection, x_offset, y_offset in tile_detections
    ]
    return non_max_suppression(translated, iou_threshold)


__all__ = ["OrientedBox", "merge_tile_detections", "non_max_suppression"]
