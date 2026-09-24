"""Tests for oriented box geometry."""

import math

import numpy as np
import pytest

from orbital_recon.ml.postprocess.geometry import OrientedBox, rotated_iou


class TestOrientedBox:
    def test_area(self) -> None:
        assert OrientedBox(0.0, 0.0, 10.0, 4.0, 0.0).area == pytest.approx(40.0)

    def test_axis_aligned_corners(self) -> None:
        corners = OrientedBox(5.0, 5.0, 4.0, 2.0, 0.0).corners()
        expected = np.array([[3.0, 4.0], [7.0, 4.0], [7.0, 6.0], [3.0, 6.0]])
        np.testing.assert_allclose(corners, expected, atol=1e-5)

    def test_rotation_preserves_area(self) -> None:
        """A rotated box keeps its area; only the enclosing bounds grow."""
        rotated = OrientedBox(0.0, 0.0, 10.0, 4.0, math.pi / 4)
        corners = rotated.corners()
        shoelace = 0.5 * abs(
            sum(
                corners[i, 0] * corners[(i + 1) % 4, 1]
                - corners[(i + 1) % 4, 0] * corners[i, 1]
                for i in range(4)
            )
        )
        assert shoelace == pytest.approx(40.0, rel=1e-4)

    def test_ninety_degree_rotation_swaps_bounds(self) -> None:
        x_min, y_min, x_max, y_max = OrientedBox(
            0.0, 0.0, 10.0, 4.0, math.pi / 2
        ).axis_aligned_bounds()
        assert (x_max - x_min) == pytest.approx(4.0, abs=1e-4)
        assert (y_max - y_min) == pytest.approx(10.0, abs=1e-4)

    def test_translated_moves_centre_only(self) -> None:
        moved = OrientedBox(1.0, 2.0, 6.0, 3.0, 0.5).translated(10.0, 20.0)
        assert (moved.cx, moved.cy) == (11.0, 22.0)
        assert (moved.width, moved.height, moved.angle) == (6.0, 3.0, 0.5)


class TestRotatedIoU:
    def test_identical_boxes(self) -> None:
        box = OrientedBox(10.0, 10.0, 8.0, 4.0, 0.3)
        assert rotated_iou(box, box) == pytest.approx(1.0, abs=1e-3)

    def test_disjoint_boxes(self) -> None:
        left = OrientedBox(0.0, 0.0, 4.0, 4.0, 0.0)
        right = OrientedBox(100.0, 100.0, 4.0, 4.0, 0.0)
        assert rotated_iou(left, right) == 0.0

    def test_half_overlap(self) -> None:
        left = OrientedBox(0.0, 0.0, 4.0, 4.0, 0.0)
        right = OrientedBox(2.0, 0.0, 4.0, 4.0, 0.0)
        # Intersection 8, union 24.
        assert rotated_iou(left, right) == pytest.approx(1.0 / 3.0, abs=1e-3)

    def test_is_symmetric(self) -> None:
        first = OrientedBox(5.0, 5.0, 10.0, 3.0, 0.4)
        second = OrientedBox(6.0, 5.5, 9.0, 4.0, 0.9)
        assert rotated_iou(first, second) == pytest.approx(rotated_iou(second, first))

    def test_perpendicular_boxes_overlap_partially(self) -> None:
        """Crossed elongated boxes overlap only where they intersect."""
        horizontal = OrientedBox(0.0, 0.0, 20.0, 2.0, 0.0)
        vertical = OrientedBox(0.0, 0.0, 20.0, 2.0, math.pi / 2)
        iou = rotated_iou(horizontal, vertical)
        # Intersection is the 2x2 centre square; union is 40 + 40 - 4.
        assert iou == pytest.approx(4.0 / 76.0, abs=1e-3)

    def test_zero_area_box_is_safe(self) -> None:
        degenerate = OrientedBox(0.0, 0.0, 0.0, 0.0, 0.0)
        assert rotated_iou(degenerate, degenerate) == 0.0
