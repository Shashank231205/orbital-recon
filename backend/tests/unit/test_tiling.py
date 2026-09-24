"""Tests for sliding-window tiling."""

import numpy as np
import pytest

from orbital_recon.ml.preprocessing.tiling import count_tiles, generate_tiles


@pytest.fixture
def scene() -> np.ndarray:
    return np.zeros((1000, 1500, 3), dtype=np.uint8)


class TestGenerateTiles:
    def test_small_image_yields_one_tile(self) -> None:
        tiles = list(generate_tiles(np.zeros((100, 100, 3), dtype=np.uint8), tile_size=640))
        assert len(tiles) == 1
        assert (tiles[0].x_offset, tiles[0].y_offset) == (0, 0)

    def test_tiles_are_square_and_full_size(self, scene: np.ndarray) -> None:
        for tile in generate_tiles(scene, tile_size=640, overlap=0.2):
            assert tile.image.shape[:2] == (640, 640)

    def test_coverage_is_complete(self, scene: np.ndarray) -> None:
        """Every pixel of the scene must appear in at least one tile."""
        covered = np.zeros(scene.shape[:2], dtype=bool)
        for tile in generate_tiles(scene, tile_size=640, overlap=0.2):
            covered[
                tile.y_offset : tile.y_offset + 640, tile.x_offset : tile.x_offset + 640
            ] = True
        assert covered.all()

    def test_indices_are_sequential(self, scene: np.ndarray) -> None:
        indices = [tile.index for tile in generate_tiles(scene, tile_size=640)]
        assert indices == list(range(len(indices)))

    def test_zero_overlap_still_covers_edges(self) -> None:
        image = np.zeros((700, 700, 3), dtype=np.uint8)
        offsets = {(t.x_offset, t.y_offset) for t in generate_tiles(image, 640, 0.0)}
        # The clamped final origin guarantees the right and bottom edges are reached.
        assert (60, 60) in offsets

    def test_count_matches_generated(self, scene: np.ndarray) -> None:
        generated = len(list(generate_tiles(scene, tile_size=640, overlap=0.2)))
        assert count_tiles(1000, 1500, 640, 0.2) == generated

    @pytest.mark.parametrize(
        ("tile_size", "overlap"),
        [(0, 0.2), (-1, 0.2), (640, 1.0), (640, -0.1)],
    )
    def test_invalid_arguments_rejected(
        self, scene: np.ndarray, tile_size: int, overlap: float
    ) -> None:
        with pytest.raises(ValueError):
            list(generate_tiles(scene, tile_size=tile_size, overlap=overlap))
