"""Model loading and device selection.

Weights are loaded once and reused for the process lifetime. Loading a detector
costs seconds and holds hundreds of megabytes of VRAM, so doing it per request
would dominate latency and quickly exhaust a laptop-class GPU.
"""

import threading
from pathlib import Path
from typing import Any

import torch

from orbital_recon.core.exceptions import ModelLoadError
from orbital_recon.core.logging import get_logger

logger = get_logger(__name__)

# Below this, CUDA context plus model weights leave too little headroom for
# activations, and inference fails mid-scene rather than at startup.
_MIN_USABLE_VRAM_BYTES = 2 * 1024**3


def resolve_device(preference: str = "auto") -> str:
    """Choose the inference device.

    Args:
        preference: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        The device string passed to the model.

    Raises:
        ModelLoadError: If CUDA is explicitly requested but unavailable.
    """
    if preference == "cpu":
        return "cpu"

    cuda_available = torch.cuda.is_available()

    if preference == "cuda":
        if not cuda_available:
            raise ModelLoadError("CUDA was requested but is not available on this host")
        return "cuda"

    if not cuda_available:
        logger.info("device_selected", device="cpu", reason="cuda_unavailable")
        return "cpu"

    vram = torch.cuda.get_device_properties(0).total_memory
    if vram < _MIN_USABLE_VRAM_BYTES:
        logger.warning(
            "device_downgraded",
            device="cpu",
            reason="insufficient_vram",
            vram_gb=round(vram / 1024**3, 2),
        )
        return "cpu"

    logger.info(
        "device_selected",
        device="cuda",
        name=torch.cuda.get_device_name(0),
        vram_gb=round(vram / 1024**3, 2),
    )
    return "cuda"


class ModelRegistry:
    """Thread-safe cache of loaded models keyed by weight path.

    FastAPI serves requests from a thread pool, so two concurrent uploads can
    race to load the same weights. The lock ensures a single load wins and the
    rest reuse it.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    def load_yolo(self, weights: str | Path, device: str) -> Any:
        """Load an Ultralytics model, reusing a cached instance when present."""
        key = f"{weights}:{device}"

        cached = self._models.get(key)
        if cached is not None:
            return cached

        with self._lock:
            # Re-check: another thread may have loaded while this one waited.
            cached = self._models.get(key)
            if cached is not None:
                return cached

            logger.info("model_loading", weights=str(weights), device=device)
            try:
                from ultralytics import YOLO

                model = YOLO(str(weights))
                model.to(device)
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load weights {weights!s}: {exc}",
                    details={"weights": str(weights), "device": device},
                ) from exc

            self._models[key] = model
            logger.info("model_loaded", weights=str(weights), device=device)
            return model

    def clear(self) -> None:
        """Release cached models and free GPU memory."""
        with self._lock:
            self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


registry = ModelRegistry()
