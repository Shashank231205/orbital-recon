"""Application configuration loaded from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    SettingsConfigDict,
)


def _default_data_dir() -> Path:
    """Locate the directory for uploads, weights and the database.

    Walking up from this module only works while the package is used from a
    source checkout. Once installed normally it lives under ``site-packages``,
    where the equivalent ancestor is somewhere inside the virtual environment,
    so writes would land in the installation rather than beside the project.

    A checkout is therefore detected explicitly, and anything else falls back to
    a directory under the working directory. Either can be overridden with
    ``DATA_DIR``.
    """
    module_path = Path(__file__).resolve()

    # <root>/backend/src/orbital_recon/core/config.py
    checkout_root = module_path.parents[4]
    if (checkout_root / "backend" / "pyproject.toml").is_file():
        return checkout_root / "data"

    return Path.cwd() / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    database_url: str = ""

    data_dir: Path = Field(default_factory=_default_data_dir)
    # Both default to positions under data_dir; see _derive_paths.
    upload_dir: Path = Path()
    model_dir: Path = Path()

    # Detection
    detector_weights: str = "yolov8s-obb.pt"
    detection_device: Literal["auto", "cuda", "cpu"] = "auto"
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    tile_size: int = Field(default=640, ge=256, le=2048)
    tile_overlap: float = Field(default=0.2, ge=0.0, lt=0.9)
    max_upload_bytes: int = 512 * 1024 * 1024

    # LLM providers, tried in the listed order.
    llm_provider_order: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["gemini", "groq"]
    )
    llm_timeout_seconds: float = 20.0
    llm_health_ttl_seconds: float = 60.0
    gemini_api_key: str | None = None
    # Alias rather than a pinned version, so the default does not go stale
    # as providers retire individual model releases.
    gemini_model: str = "gemini-flash-latest"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

    @field_validator("cors_origins", "llm_provider_order", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Parse comma-separated list values.

        These fields are marked ``NoDecode`` because pydantic-settings would
        otherwise try to JSON-decode any list-typed field before validators
        run, which rejects the plain ``a,b`` form that is natural in a .env
        file.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        """Fill in paths that default to positions under ``data_dir``.

        Deriving them here rather than at field definition means overriding
        ``DATA_DIR`` alone moves the uploads, weights and database together,
        instead of silently leaving them behind at the old location.
        """
        if self.upload_dir == Path():
            self.upload_dir = self.data_dir / "raw"
        if self.model_dir == Path():
            self.model_dir = self.data_dir / "models"
        if not self.database_url:
            # An absolute path, so the database does not move with the working
            # directory the process happened to start in.
            self.database_url = (
                f"sqlite+aiosqlite:///{(self.data_dir / 'orbital_recon.db').as_posix()}"
            )
        return self

    def ensure_directories(self) -> None:
        """Create the directories the service writes into."""
        for directory in (self.data_dir, self.upload_dir, self.model_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
