"""Tests for scene ingestion."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from orbital_recon.core.exceptions import (
    PayloadTooLargeError,
    UnsupportedMediaError,
    ValidationError,
)
from orbital_recon.schemas.enums import Modality
from orbital_recon.services.scenes import (
    build_scene,
    read_dimensions,
    store_upload,
    validate_upload,
)


def png_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestValidateUpload:
    @pytest.mark.parametrize("name", ["a.png", "a.PNG", "b.tif", "c.tiff", "d.jpg", "e.jpeg"])
    def test_accepts_supported_formats(self, name: str) -> None:
        assert validate_upload(name, 100, 1000) == Path(name).suffix.lower()

    def test_rejects_unsupported_format(self) -> None:
        with pytest.raises(UnsupportedMediaError):
            validate_upload("notes.txt", 100, 1000)

    def test_rejects_missing_filename(self) -> None:
        with pytest.raises(ValidationError):
            validate_upload("   ", 100, 1000)

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValidationError):
            validate_upload("a.png", 0, 1000)

    def test_rejects_oversized_payload(self) -> None:
        with pytest.raises(PayloadTooLargeError):
            validate_upload("a.png", 2000, 1000)


class TestStoreUpload:
    def test_writes_content_under_generated_name(self, tmp_path: Path) -> None:
        path = store_upload(b"data", ".png", tmp_path)

        assert path.read_bytes() == b"data"
        assert path.suffix == ".png"
        assert path.stem != "data"

    def test_repeated_uploads_do_not_collide(self, tmp_path: Path) -> None:
        first = store_upload(b"one", ".png", tmp_path)
        second = store_upload(b"two", ".png", tmp_path)

        assert first != second
        assert first.read_bytes() == b"one"


class TestReadDimensions:
    def test_reads_true_dimensions(self, tmp_path: Path) -> None:
        path = tmp_path / "scene.png"
        path.write_bytes(png_bytes(321, 123))

        assert read_dimensions(path) == (321, 123)

    def test_error_names_the_uploaded_file(self, tmp_path: Path) -> None:
        """Errors quote the original name, not the generated storage name."""
        path = tmp_path / "a1b2c3.png"
        path.write_bytes(b"not an image")

        with pytest.raises(ValidationError, match="harbour-survey.png"):
            read_dimensions(path, "harbour-survey.png")


class TestBuildScene:
    def test_records_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "stored.png"
        content = png_bytes(200, 150)
        path.write_bytes(content)

        scene = build_scene("survey.png", path, Modality.ELECTRO_OPTICAL, len(content))

        assert scene.filename == "survey.png"
        assert (scene.width, scene.height) == (200, 150)
        assert scene.is_georeferenced is False
        assert scene.crs is None

    def test_removes_stored_file_when_unreadable(self, tmp_path: Path) -> None:
        """A rejected upload must not stay on disk."""
        path = tmp_path / "stored.png"
        path.write_bytes(b"not an image")

        with pytest.raises(ValidationError):
            build_scene("broken.png", path, Modality.ELECTRO_OPTICAL, 12)

        assert not path.exists()
