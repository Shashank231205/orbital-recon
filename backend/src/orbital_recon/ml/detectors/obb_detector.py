"""Oriented bounding box detector built on Ultralytics YOLO-OBB."""

from collections.abc import Callable, Sequence

import numpy as np
import torch

from orbital_recon.core.exceptions import InferenceError
from orbital_recon.core.logging import get_logger
from orbital_recon.ml.postprocess.geometry import OrientedBox
from orbital_recon.ml.preprocessing.tiling import Tile, count_tiles, generate_tiles
from orbital_recon.ml.registry.loader import registry, resolve_device
from orbital_recon.schemas.detection import RawDetection

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int], None]


class ObbDetector:
    """Detects oriented targets in large overhead scenes.

    The scene is tiled, each batch of tiles is inferred on, and detections are
    returned in tile-local coordinates paired with their tile offsets. Merging
    into scene coordinates is left to the caller so that suppression policy
    stays outside the model wrapper.
    """

    def __init__(
        self,
        weights: str,
        device: str = "auto",
        confidence_threshold: float = 0.25,
        tile_size: int = 640,
        tile_overlap: float = 0.2,
        batch_size: int = 4,
    ) -> None:
        self.weights = weights
        self.device = resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self._model = registry.load_yolo(weights, self.device)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._model.names)

    def _infer_batch(self, tiles: Sequence[Tile]) -> list[list[RawDetection]]:
        """Run the model over one batch, halving the batch on GPU exhaustion.

        A 4 GB laptop GPU can exhaust VRAM on a dense batch. Rather than failing
        the whole scene, the batch is split and retried; a single tile that still
        cannot fit is a genuine error.
        """
        images = [tile.image for tile in tiles]

        try:
            results = self._model.predict(
                images,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError:
            if len(tiles) == 1:
                raise InferenceError(
                    "GPU ran out of memory on a single tile; reduce tile_size"
                ) from None

            torch.cuda.empty_cache()
            midpoint = len(tiles) // 2
            logger.warning("batch_split_after_oom", original_size=len(tiles))
            return self._infer_batch(tiles[:midpoint]) + self._infer_batch(tiles[midpoint:])

        return [self._parse_result(result) for result in results]

    def _parse_result(self, result: object) -> list[RawDetection]:
        """Convert one Ultralytics result into detections."""
        obb = getattr(result, "obb", None)
        if obb is None or obb.xywhr is None or len(obb.xywhr) == 0:
            return []

        boxes = obb.xywhr.cpu().numpy()
        confidences = obb.conf.cpu().numpy()
        class_ids = obb.cls.cpu().numpy().astype(int)
        names = self.class_names

        return [
            RawDetection(
                box=OrientedBox(
                    cx=float(box[0]),
                    cy=float(box[1]),
                    width=float(box[2]),
                    height=float(box[3]),
                    angle=float(box[4]),
                ),
                class_name=names.get(int(class_id), str(class_id)),
                confidence=float(confidence),
            )
            for box, confidence, class_id in zip(boxes, confidences, class_ids, strict=True)
        ]

    def detect(
        self,
        image: np.ndarray,
        on_progress: ProgressCallback | None = None,
    ) -> list[tuple[RawDetection, int, int]]:
        """Detect targets across a full scene.

        Args:
            image: Scene as an ``(H, W, 3)`` uint8 array.
            on_progress: Called with ``(tiles_done, tiles_total)`` after each batch.

        Returns:
            Triples of ``(detection, x_offset, y_offset)`` in tile-local coordinates.

        Raises:
            InferenceError: If inference fails irrecoverably.
        """
        height, width = image.shape[:2]
        total_tiles = count_tiles(height, width, self.tile_size, self.tile_overlap)
        logger.info("detection_started", width=width, height=height, tiles=total_tiles)

        collected: list[tuple[RawDetection, int, int]] = []
        batch: list[Tile] = []
        processed = 0

        def flush(current: list[Tile]) -> None:
            nonlocal processed
            if not current:
                return
            for tile, detections in zip(current, self._infer_batch(current), strict=True):
                collected.extend(
                    (detection, tile.x_offset, tile.y_offset) for detection in detections
                )
            processed += len(current)
            if on_progress is not None:
                on_progress(processed, total_tiles)

        try:
            for tile in generate_tiles(image, self.tile_size, self.tile_overlap):
                batch.append(tile)
                if len(batch) >= self.batch_size:
                    flush(batch)
                    batch = []
            flush(batch)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc

        logger.info("detection_finished", raw_detections=len(collected), tiles=total_tiles)
        return collected
