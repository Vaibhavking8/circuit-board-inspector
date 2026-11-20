"""
comparison/edge.py — edge/gradient similarity signal for per-part comparison.

Uses Sobel gradient magnitudes on both reference and inspection crops, then
computes normalised cross-correlation (NCC) of the gradient maps.

Gradient magnitude is illumination-invariant (brightness offsets cancel).
Avoids fragile Canny thresholds — raw magnitude is used throughout.
No web-framework, storage, or imaging imports.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from boardinspect.comparison.interface import SignalResult
from boardinspect.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

_SOBEL_KSIZE = 3  # fixed Sobel kernel; not configurable (standard choice)
_MIN_STD = 1e-6   # prevent division by zero in NCC


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude of a grayscale image.

    Args:
        image: Grayscale uint8 or float32 ndarray (H×W).

    Returns:
        float32 gradient magnitude array (H×W), not normalised.
    """
    img_f = image.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=_SOBEL_KSIZE)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=_SOBEL_KSIZE)
    return np.sqrt(gx * gx + gy * gy)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation between two arrays of the same shape.

    Returns a float in [-1, 1]. Returns 0.0 if either array is flat
    (std ≈ 0), which avoids division by zero for featureless regions.
    """
    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std()
    b_std = b.std()

    if a_std < _MIN_STD or b_std < _MIN_STD:
        # One or both arrays are essentially constant; NCC is undefined.
        # Return 0.0 (neutral, neither strong presence nor absence signal).
        return 0.0

    a_norm = (a - a_mean) / a_std
    b_norm = (b - b_mean) / b_std
    return float(np.mean(a_norm * b_norm))


class EdgeGradientSignal:
    """Edge/gradient similarity between a reference crop and an inspection region.

    Implements the Comparator protocol. Strategy:
    1. Compute Sobel gradient magnitude on the reference crop.
    2. Run template matching on the gradient magnitude maps to find the
       best-matching location within the search region (same approach as
       the template signal but applied to gradient maps).
    3. Extract the gradient crop at the best location and compute NCC
       against the reference gradient.

    The final score is in [-1, 1] (NCC range), where:
      ≈ 1.0  → edge pattern matches well (part likely Present)
      ≈ 0.0  → flat region, insufficient gradient information
      < 0.0  → edge patterns anti-correlated (unusual, treated as absent-like)
    """

    @property
    def name(self) -> str:
        return "edge"

    def compare(
        self,
        ref_crop: np.ndarray,
        search_region: np.ndarray,
        config: Config = DEFAULT_CONFIG,
    ) -> SignalResult:
        """Compute gradient similarity between *ref_crop* and *search_region*.

        Finds the best matching location within *search_region* using
        gradient magnitude template matching, then computes NCC at that
        location.

        Args:
            ref_crop:      Grayscale reference crop (uint8, H×W).
            search_region: Grayscale padded search window (uint8, H'×W').
                           H' ≥ H and W' ≥ W.
            config:        Config instance (not currently used for edge params;
                           included for protocol compliance).

        Returns:
            SignalResult with name='edge', value in [-1, 1], and location
            as (dx, dy) offset within the search window.
        """
        # Compute gradient magnitudes
        ref_grad = _gradient_magnitude(ref_crop)
        region_grad = _gradient_magnitude(search_region)

        ref_h, ref_w = ref_grad.shape[:2]
        region_h, region_w = region_grad.shape[:2]

        # If search region is smaller than ref crop (shouldn't happen but be safe)
        if region_h < ref_h or region_w < ref_w:
            logger.warning(
                "EdgeGradient: search region (%dx%d) smaller than ref crop (%dx%d); "
                "computing NCC on full region.",
                region_w, region_h, ref_w, ref_h,
            )
            # Resize region to match crop for a direct NCC
            resized = cv2.resize(region_grad, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
            score = _ncc(ref_grad, resized)
            return SignalResult(name=self.name, value=score, location=(0, 0))

        # Template match on gradient maps to find best location.
        # Normalise to [0, 255] uint8 for matchTemplate.
        def _normalise_u8(arr: np.ndarray) -> np.ndarray:
            mx = arr.max()
            if mx < 1e-9:
                return np.zeros_like(arr, dtype=np.uint8)
            return (arr * 255.0 / mx).astype(np.uint8)

        ref_u8 = _normalise_u8(ref_grad)
        region_u8 = _normalise_u8(region_grad)

        result_map = cv2.matchTemplate(region_u8, ref_u8, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result_map)
        best_x, best_y = max_loc  # top-left of best match in search_region

        # Extract the aligned crop at best location and compute NCC on gradients
        best_crop_grad = region_grad[best_y: best_y + ref_h, best_x: best_x + ref_w]
        score = _ncc(ref_grad, best_crop_grad)

        # Location as offset from the expected centre of the search window
        expected_x = (region_w - ref_w) // 2
        expected_y = (region_h - ref_h) // 2
        dx = best_x - expected_x
        dy = best_y - expected_y

        logger.debug(
            "EdgeGradient: score=%.3f, best_loc=(%d, %d), offset=(%d, %d).",
            score, best_x, best_y, dx, dy,
        )

        return SignalResult(name=self.name, value=score, location=(dx, dy))


# Singleton for use by the pipeline
EDGE_GRADIENT_SIGNAL = EdgeGradientSignal()
