"""
registration/cascade.py — cascade of alignment strategies with validation.

Runs strategies in config order:
  1. skip_if_aligned  — phase-correlation check; if already aligned, return Identity.
  2. ecc_euclidean    — ECC with MOTION_EUCLIDEAN (rotation + translation).
  3. ecc_affine       — ECC with MOTION_AFFINE (adds scale + shear).
  4. orb_homography   — ORB features + RANSAC homography fallback.

Each failure (typed error or low quality) is caught and the next strategy tried.
Transform validation rejects degenerate solutions (extreme translation, scale, rotation).
If all strategies fail or produce low-quality results, returns RegistrationResult.failed().
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.errors import AlignmentError
from boardinspect.registration.ecc import ECC_AFFINE, ECC_EUCLIDEAN
from boardinspect.registration.interface import RegistrationResult, RegistrationStatus
from boardinspect.registration.orb import ORB_REGISTRAR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skip-if-aligned helper
# ---------------------------------------------------------------------------

def _is_already_aligned(
    ref_gray: np.ndarray,
    test_gray: np.ndarray,
    tolerance_px: float,
) -> bool:
    """Return True if the two images are already well-aligned.

    Uses phase correlation to estimate the dominant translation between the
    images. If the magnitude of the translation vector is below *tolerance_px*,
    the images are considered already aligned.

    Args:
        ref_gray:      Reference grayscale image (uint8, H×W).
        test_gray:     Test grayscale image (uint8, H×W).
        tolerance_px:  Max displacement in pixels to consider aligned.

    Returns:
        True if estimated translation < tolerance_px.
    """
    h, w = ref_gray.shape[:2]
    # Resize to a small working size for speed
    small_size = 256
    scale = small_size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    ref_small = cv2.resize(ref_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    test_small = cv2.resize(test_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ref_f = ref_small.astype(np.float32)
    test_f = test_small.astype(np.float32)

    shift, _ = cv2.phaseCorrelate(ref_f, test_f)
    # Scale shift back to original image coordinates
    shift_original = (shift[0] / scale, shift[1] / scale)
    displacement = math.sqrt(shift_original[0] ** 2 + shift_original[1] ** 2)
    logger.debug(
        "Skip-if-aligned: shift=(%.2f, %.2f) px (original scale), tolerance=%.2f px.",
        shift_original[0], shift_original[1], tolerance_px,
    )
    return displacement < tolerance_px


# ---------------------------------------------------------------------------
# Transform validation
# ---------------------------------------------------------------------------

def _extract_affine_params(
    transform: np.ndarray,
) -> tuple[float, float, float, float]:
    """Extract (tx, ty, scale, rotation_degrees) from a 2×3 or 3×3 matrix."""
    if transform.shape == (3, 3):
        # Approximate affine params from homography
        a = transform[0, 0]
        b = transform[0, 1]
        tx = transform[0, 2]
        ty = transform[1, 2]
    else:
        a, b, tx = transform[0]
        _, _, ty = transform[1]

    scale = math.sqrt(a * a + b * b)  # scale of the x-axis
    rotation_rad = math.atan2(b, a)
    rotation_deg = math.degrees(rotation_rad)
    return tx, ty, scale, rotation_deg


def _validate_transform(
    transform: np.ndarray,
    image_width: int,
    image_height: int,
    config: Config,
) -> bool:
    """Return True if the transform is within acceptable bounds.

    Rejects transforms with:
    - Translation > max_translation_fraction × image size
    - Scale outside [min_scale, max_scale]
    - Rotation > max_rotation_degrees
    """
    reg = config.registration
    tx, ty, scale, rot_deg = _extract_affine_params(transform)

    max_tx = reg.transform_max_translation_fraction * image_width
    max_ty = reg.transform_max_translation_fraction * image_height

    if abs(tx) > max_tx or abs(ty) > max_ty:
        logger.warning(
            "Transform validation FAILED: translation (%.1f, %.1f) exceeds limits "
            "(%.1f, %.1f).", tx, ty, max_tx, max_ty,
        )
        return False

    if scale < reg.transform_min_scale or scale > reg.transform_max_scale:
        logger.warning(
            "Transform validation FAILED: scale %.3f outside [%.2f, %.2f].",
            scale, reg.transform_min_scale, reg.transform_max_scale,
        )
        return False

    if abs(rot_deg) > reg.transform_max_rotation_degrees:
        logger.warning(
            "Transform validation FAILED: rotation %.1f° exceeds ±%.1f°.",
            rot_deg, reg.transform_max_rotation_degrees,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "ecc_euclidean":  ECC_EUCLIDEAN,
    "ecc_affine":     ECC_AFFINE,
    "orb_homography": ORB_REGISTRAR,
}


class RegistrationCascade:
    """Run alignment strategies in config order, returning the first success."""

    def register(
        self,
        ref_gray: np.ndarray,
        test_gray: np.ndarray,
        config: Config = DEFAULT_CONFIG,
    ) -> RegistrationResult:
        """Align *test_gray* to *ref_gray* using the configured strategy cascade.

        Args:
            ref_gray:  Reference grayscale image (uint8, H×W).
            test_gray: Test grayscale image (uint8, H×W).
            config:    Config instance with cascade_order and all strategy params.

        Returns:
            RegistrationResult. Status is Identity, Aligned, or Failed.
            Never raises — all strategy errors are caught internally.
        """
        h, w = ref_gray.shape[:2]
        reg = config.registration

        for strategy_name in reg.cascade_order:

            # --- Skip-if-aligned (special case, no external registrar) ---
            if strategy_name == "skip_if_aligned":
                try:
                    if _is_already_aligned(ref_gray, test_gray, reg.skip_if_aligned_tolerance_px):
                        logger.debug("Cascade: skip_if_aligned triggered → Identity.")
                        return RegistrationResult.identity()
                except Exception as exc:
                    logger.debug("skip_if_aligned check failed: %s. Continuing.", exc)
                continue

            # --- Registrar strategies ---
            registrar = _STRATEGY_MAP.get(strategy_name)
            if registrar is None:
                logger.warning("Cascade: unknown strategy '%s'; skipping.", strategy_name)
                continue

            try:
                result = registrar.register(ref_gray, test_gray, config)
            except AlignmentError as exc:
                logger.debug(
                    "Cascade: %s raised %s ('%s'); trying next strategy.",
                    strategy_name, type(exc).__name__, exc.message,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Cascade: %s raised unexpected error %s; trying next strategy.",
                    strategy_name, exc,
                )
                continue

            # Reject low-quality results
            if result.status == RegistrationStatus.Failed:
                logger.debug(
                    "Cascade: %s quality %.3f below minimum %.3f; trying next.",
                    strategy_name, result.quality, reg.min_alignment_quality,
                )
                continue

            # Reject degenerate transforms
            if not _validate_transform(result.transform, w, h, config):
                logger.debug("Cascade: %s transform failed validation; trying next.", strategy_name)
                continue

            logger.debug(
                "Cascade: %s succeeded (quality=%.3f, transform_type=%s).",
                strategy_name, result.quality, result.transform_type,
            )
            return result

        # All strategies exhausted
        logger.warning("Cascade: all strategies failed or produced degenerate transforms.")
        return RegistrationResult.failed("cascade")


# Singleton
REGISTRATION_CASCADE = RegistrationCascade()
