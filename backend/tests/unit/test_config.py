"""Tests for settings resolution."""

from pathlib import Path

import pytest

from orbital_recon.core.config import Settings, _default_data_dir


class TestDataDirectory:
    def test_resolves_to_project_data_in_a_checkout(self) -> None:
        """Running from source keeps data beside the repository."""
        resolved = _default_data_dir()

        assert resolved.name == "data"
        assert (resolved.parent / "backend" / "pyproject.toml").is_file()

    def test_falls_back_to_working_directory_when_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An installed package must not write inside its own installation.

        Walking up from the module lands in site-packages once the package is
        installed normally, so the checkout layout is verified before it is
        trusted.
        """
        installed = tmp_path / "site-packages" / "orbital_recon" / "core" / "config.py"
        installed.parent.mkdir(parents=True)
        installed.touch()

        monkeypatch.setattr("orbital_recon.core.config.__file__", str(installed))
        monkeypatch.chdir(tmp_path)

        assert _default_data_dir() == tmp_path / "data"


class TestDerivedPaths:
    def test_defaults_sit_under_data_dir(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path)

        assert settings.upload_dir == tmp_path / "raw"
        assert settings.model_dir == tmp_path / "models"

    def test_database_url_is_absolute(self, tmp_path: Path) -> None:
        """The database must not move with the process working directory."""
        settings = Settings(data_dir=tmp_path)

        assert settings.database_url.startswith("sqlite+aiosqlite:///")
        assert "orbital_recon.db" in settings.database_url
        assert "./" not in settings.database_url

    def test_explicit_paths_are_respected(self, tmp_path: Path) -> None:
        settings = Settings(
            data_dir=tmp_path,
            upload_dir=tmp_path / "elsewhere",
            database_url="sqlite+aiosqlite:///:memory:",
        )

        assert settings.upload_dir == tmp_path / "elsewhere"
        assert settings.database_url == "sqlite+aiosqlite:///:memory:"

    def test_ensure_directories_creates_all_three(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path / "fresh")
        settings.ensure_directories()

        assert settings.data_dir.is_dir()
        assert settings.upload_dir.is_dir()
        assert settings.model_dir.is_dir()


class TestListParsing:
    def test_comma_separated_values_parse(self) -> None:
        """A .env file cannot carry JSON, so plain lists must be accepted."""
        settings = Settings(llm_provider_order="groq,gemini")  # type: ignore[arg-type]

        assert settings.llm_provider_order == ["groq", "gemini"]

    def test_whitespace_is_trimmed(self) -> None:
        settings = Settings(cors_origins=" http://a , http://b ")  # type: ignore[arg-type]

        assert settings.cors_origins == ["http://a", "http://b"]
