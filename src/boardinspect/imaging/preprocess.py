"""
imaging/preprocess.py — image preprocessing pipeline for BoardInspect.

All functions are pure: they accept numpy arrays and return numpy arrays.
No file I/O, no network, no web-framework imports.
Every parameter is read from the Config object; no magic numbers.
"""
from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

Stage = Literal["registration", "comparison"]


def to_working(
    image: np.ndarray,
    longest_side: int,
) -> np.ndarray:
    """Resize *image* so its longest side equals *longest_side*.

    Uses cv2.INTER_AREA for downscaling and cv2.INTER_LINEAR for upscaling
    (upscaling is unusual but handled gracefully). Aspect ratio is preserved.
    If the image already fits, returns it unchanged (no copy).

    Args:
        image:        BGR or grayscale ndarray.
        longest_side: Target length for the longest dimension in pixels.

    Returns:
        Resized ndarray (same dtype as input).
    """
    h, w = image.shape[:2]
    current_longest = max(h, w)
    if current_longest == longest_side:
        return image

    scale = longest_side / current_longest
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    logger.debug(
        "Resized %dx%d → %dx%d (scale=%.3f, interp=%s).",
        w, h, new_w, new_h, scale,
        "INTER_AREA" if scale < 1.0 else "INTER_LINEAR",
    )
    return resized


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale.

    If the image is already single-channel, returns it unchanged.

    Args:
        image: BGR uint8 ndarray (H×W×3) or grayscale (H×W) or (H×W×1).

    Returns:
        Grayscale uint8 ndarray (H×W).
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a mild Gaussian blur with the given *kernel_size* (must be odd).

    Args:
        image:       Grayscale or BGR ndarray.
        kernel_size: Odd integer kernel size (e.g. 3).

    Returns:
        Blurred ndarray (same dtype and shape as input).
    """
    if kernel_size <= 1:
        return image
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0)


def apply_clahe(image: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
    """Apply CLAHE histogram equalisation.

    Must be applied identically to reference and inspection images to avoid
    introducing asymmetric changes. Applied to grayscale only; for colour
    images the caller must convert first.

    Args:
        image:      Grayscale uint8 ndarray (H×W).
        clip_limit: CLAHE clip limit (e.g. 2.0).
        grid_size:  Tile grid size in pixels; tiles are square (e.g. 8).

    Returns:
        Equalised grayscale ndarray (same shape).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    return clahe.apply(image)


def preprocess(
    image: np.ndarray,
    config: Config = DEFAULT_CONFIG,
    stage: Stage = "registration",
) -> np.ndarray:
    """Full preprocessing pipeline for one stage.

    Steps (each conditional on config):
    1. Resize to working resolution for the given stage.
    2. Convert to grayscale (if config.preprocessing.grayscale).
    3. Apply Gaussian blur (if config.preprocessing.gaussian_blur).
    4. Apply CLAHE (if config.preprocessing.clahe). [EXPERIMENT]

    Args:
        image:  Input BGR ndarray.
        config: Config instance.
        stage:  ``"registration"`` or ``"comparison"``; selects resolution.

    Returns:
        Preprocessed ndarray (grayscale H×W or BGR H×W×3 depending on config).
    """
    reg = config.registration
    comp = config.comparison
    pre = config.preprocessing

    longest_side = reg.longest_side if stage == "registration" else comp.longest_side
    result = to_working(image, longest_side)

    if pre.grayscale:
        result = to_grayscale(result)

    if pre.gaussian_blur:
        result = apply_blur(result, pre.gaussian_kernel_size)

    if pre.clahe:
        if result.ndim != 2:
            result = to_grayscale(result)
        result = apply_clahe(result, pre.clahe_clip_limit, pre.clahe_grid_size)

    logger.debug(
        "preprocess(stage=%s): output shape=%s dtype=%s.",
        stage, result.shape, result.dtype,
    )
    return result
