"""Oriented bounding box geometry.

Overhead imagery has no canonical object orientation: an aircraft on a taxiway
or a ship at anchor can sit at any angle. An axis-aligned box around a diagonal
object is mostly background, which inflates overlap between neighbouring
objects and makes axis-aligned suppression merge distinct targets. Oriented
boxes, stored as ``(cx, cy, w, h, angle)``, keep the geometry tight.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class OrientedBox:
    """A rotated rectangle in image pixel coordinates.

    Attributes:
        cx: Centre x in pixels.
        cy: Centre y in pixels.
        width: Extent along the rotated x axis.
        height: Extent along the rotated y axis.
        angle: Rotation in radians, counter-clockwise from the image x axis.
    """

    cx: float
    cy: float
    width: float
    height: float
    angle: float

    @property
    def area(self) -> float:
        return self.width * self.height

    def corners(self) -> np.ndarray:
        """Return the four corners as a ``(4, 2)`` array in clockwise order."""
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        half_w, half_h = self.width / 2.0, self.height / 2.0

        local = np.array(
            [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
            dtype=np.float32,
        )
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        return local @ rotation.T + np.array([self.cx, self.cy], dtype=np.float32)

    def translated(self, dx: float, dy: float) -> "OrientedBox":
        """Return a copy shifted by ``(dx, dy)``, used to map tiles into the scene."""
        return OrientedBox(self.cx + dx, self.cy + dy, self.width, self.height, self.angle)

    def axis_aligned_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(x_min, y_min, x_max, y_max)`` enclosing the rotated box."""
        corners = self.corners()
        return (
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
        )


def rotated_iou(first: OrientedBox, second: OrientedBox) -> float:
    """Intersection over union of two oriented boxes.

    Uses exact polygon clipping rather than an axis-aligned approximation,
    because the approximation systematically overestimates overlap for elongated
    diagonal objects such as ships and runways.
    """
    combined = first.area + second.area
    if combined <= 0.0:
        return 0.0

    retval, region = cv2.rotatedRectangleIntersection(
        ((first.cx, first.cy), (first.width, first.height), np.degrees(first.angle)),
        ((second.cx, second.cy), (second.width, second.height), np.degrees(second.angle)),
    )
    if retval == cv2.INTERSECT_NONE or region is None:
        return 0.0

    intersection = float(cv2.contourArea(cv2.convexHull(region)))
    union = combined - intersection
    return intersection / union if union > 0.0 else 0.0
