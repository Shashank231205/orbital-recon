"""Modality-specific image enhancement.

Overhead sensors produce very different statistics, so each modality gets the
correction it actually needs rather than a shared generic pipeline:

* Electro-optical scenes suffer from haze and low local contrast.
* Infrared arrives in wide bit depths dominated by outliers.
* Synthetic-aperture radar carries multiplicative speckle noise that ordinary
  smoothing filters blur straight through.
"""

import cv2
import numpy as np

from orbital_recon.schemas.enums import Modality


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    Applied to the luminance channel only so colour relationships survive, which
    matters because downstream classification uses colour cues.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def percentile_stretch(
    image: np.ndarray,
    lower: float = 2.0,
    upper: float = 98.0,
) -> np.ndarray:
    """Rescale to 8-bit using percentile bounds.

    Percentiles rather than min/max because a handful of saturated or dead
    pixels would otherwise compress the entire useful dynamic range.
    """
    low, high = np.percentile(image, (lower, upper))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    stretched = (image.astype(np.float32) - low) / (high - low)
    return (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8)


def lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Lee adaptive speckle filter for SAR imagery.

    Speckle in SAR is multiplicative, so averaging filters erase edges along with
    the noise. The Lee filter weights each pixel by local signal variance: it
    smooths aggressively in homogeneous regions and backs off near edges, which
    preserves the hard structural returns that identify man-made objects.
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    source = image.astype(np.float32)
    kernel = (window_size, window_size)

    local_mean = cv2.blur(source, kernel)
    local_sq_mean = cv2.blur(source**2, kernel)
    local_variance = np.clip(local_sq_mean - local_mean**2, 0.0, None)

    # Global coefficient of variation estimates the speckle level in the scene.
    overall_variance = float(np.var(source))
    if overall_variance <= 0.0:
        return image

    weights = local_variance / (local_variance + overall_variance)
    filtered = local_mean + weights * (source - local_mean)
    return np.clip(filtered, 0.0, 255.0).astype(np.uint8)


def preprocess(image: np.ndarray, modality: Modality) -> np.ndarray:
    """Route an image through the pipeline matching its sensor modality."""
    if modality is Modality.SAR:
        grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        despeckled = lee_filter(percentile_stretch(grey))
        return cv2.cvtColor(despeckled, cv2.COLOR_GRAY2RGB)

    if modality is Modality.INFRARED:
        grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalised = apply_clahe(percentile_stretch(grey))
        return cv2.cvtColor(equalised, cv2.COLOR_GRAY2RGB)

    prepared = image if image.dtype == np.uint8 else percentile_stretch(image)
    return apply_clahe(prepared)
