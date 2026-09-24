"""Tests for rotated non-maximum suppression."""

from orbital_recon.ml.postprocess.geometry import OrientedBox
from orbital_recon.ml.postprocess.nms import merge_tile_detections, non_max_suppression
from orbital_recon.schemas.detection import RawDetection


def make_detection(
    cx: float, cy: float, confidence: float, class_name: str = "aircraft"
) -> RawDetection:
    return RawDetection(
        box=OrientedBox(cx, cy, 10.0, 10.0, 0.0),
        class_name=class_name,
        confidence=confidence,
    )


class TestNonMaxSuppression:
    def test_empty_input(self) -> None:
        assert non_max_suppression([]) == []

    def test_keeps_highest_confidence_duplicate(self) -> None:
        kept = non_max_suppression(
            [make_detection(0.0, 0.0, 0.7), make_detection(1.0, 1.0, 0.9)],
            iou_threshold=0.45,
        )
        assert len(kept) == 1
        assert kept[0].confidence == 0.9

    def test_distinct_objects_both_survive(self) -> None:
        kept = non_max_suppression(
            [make_detection(0.0, 0.0, 0.9), make_detection(500.0, 500.0, 0.8)]
        )
        assert len(kept) == 2

    def test_different_classes_are_not_suppressed(self) -> None:
        """A vehicle beside a radar shares a footprint but is a separate target."""
        kept = non_max_suppression(
            [
                make_detection(0.0, 0.0, 0.9, "vehicle"),
                make_detection(0.0, 0.0, 0.8, "radar"),
            ]
        )
        assert len(kept) == 2
        assert {d.class_name for d in kept} == {"vehicle", "radar"}

    def test_output_sorted_by_confidence(self) -> None:
        kept = non_max_suppression(
            [
                make_detection(0.0, 0.0, 0.5),
                make_detection(200.0, 200.0, 0.95),
                make_detection(400.0, 400.0, 0.75),
            ]
        )
        assert [d.confidence for d in kept] == [0.95, 0.75, 0.5]


class TestMergeTileDetections:
    def test_offsets_applied_to_scene_coordinates(self) -> None:
        merged = merge_tile_detections([(make_detection(10.0, 10.0, 0.9), 640, 1280)])
        assert (merged[0].box.cx, merged[0].box.cy) == (650.0, 1290.0)

    def test_duplicate_across_tile_seam_collapses(self) -> None:
        """The same object seen in two overlapping tiles yields one detection."""
        merged = merge_tile_detections(
            [
                (make_detection(600.0, 100.0, 0.85), 0, 0),
                (make_detection(88.0, 100.0, 0.91), 512, 0),
            ]
        )
        assert len(merged) == 1
        assert merged[0].confidence == 0.91

    def test_empty_input(self) -> None:
        assert merge_tile_detections([]) == []
