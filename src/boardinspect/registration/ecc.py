"""
registration/ecc.py — ECC (Enhanced Correlation Coefficient) alignment strategies.

Two motion models are supported:
  - MOTION_EUCLIDEAN (rotation + translation only) — the preferred first try.
  - MOTION_AFFINE    (rotation, translation, scale, shear) — richer model, slower.

Both use cv2.findTransformECC with convergence criteria read from Config.
Quality is computed with cv2.computeECC on the result.
NonConvergenceError is raised on cv2.error (ECC did not converge).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.errors import NonConvergenceError
from boardinspect.models import TransformType
from boardinspect.registration.interface import RegistrationResult, RegistrationStatus

logger = logging.getLogger(__name__)


def _termination_criteria(config: Config) -> tuple[int, int, float]:
    """Return OpenCV termination criteria tuple from Config."""
    return (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.registration.ecc_max_iterations,
        config.registration.ecc_epsilon,
    )


def _compute_quality(
    ref_gray: np.ndarray,
    test_gray: np.ndarray,
    transform: np.ndarray,
    motion_type: int,
) -> float:
    """Compute ECC quality score of *transform* by warping and correlating.

    Warps the test image into the reference frame, then measures the ECC
    coefficient between the two. Returns a float in [0, 1]. Falls back to
    0.0 on any cv2 error.
    """
    try:
        h, w = ref_gray.shape[:2]
        if motion_type in (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE,
                           cv2.MOTION_TRANSLATION):
            warped = cv2.warpAffine(
                test_gray, transform, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
        else:
            warped = cv2.warpPerspective(
                test_gray, transform, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
        # computeECC returns the ECC coefficient in [-1, 1]; normalise to [0, 1]
        ecc = cv2.computeECC(
            ref_gray.astype(np.float32),
            warped.astype(np.float32),
        )
        return float(max(0.0, min(1.0, (ecc + 1.0) / 2.0)))
    except cv2.error as exc:
        logger.debug("Quality computation failed: %s", exc)
        return 0.0


class ECCRegistrar:
    """Align two grayscale images using the ECC criterion.

    Args:
        motion_type: ``cv2.MOTION_EUCLIDEAN`` or ``cv2.MOTION_AFFINE``.
            ``cv2.MOTION_TRANSLATION`` and ``cv2.MOTION_HOMOGRAPHY`` are also
            accepted so the class can serve as a general ECC wrapper.
    """

    def __init__(self, motion_type: int) -> None:
        if motion_type not in (
            cv2.MOTION_EUCLIDEAN,
            cv2.MOTION_AFFINE,
            cv2.MOTION_TRANSLATION,
            cv2.MOTION_HOMOGRAPHY,
        ):
            raise ValueError(f"Unsupported motion_type: {motion_type}")
        self._motion_type = motion_type

    @property
    def _transform_type(self) -> TransformType:
        return {
            cv2.MOTION_TRANSLATION: TransformType.Translation,
            cv2.MOTION_EUCLIDEAN:   TransformType.Euclidean,
            cv2.MOTION_AFFINE:      TransformType.Affine,
            cv2.MOTION_HOMOGRAPHY:  TransformType.Homography,
        }[self._motion_type]

    @property
    def _method_name(self) -> str:
        return {
            cv2.MOTION_TRANSLATION: "ecc_translation",
            cv2.MOTION_EUCLIDEAN:   "ecc_euclidean",
            cv2.MOTION_AFFINE:      "ecc_affine",
            cv2.MOTION_HOMOGRAPHY:  "ecc_homography",
        }[self._motion_type]

    def register(
        self,
        ref_gray: np.ndarray,
        test_gray: np.ndarray,
        config: Config = DEFAULT_CONFIG,
    ) -> RegistrationResult:
        """Align *test_gray* to *ref_gray* using ECC.

        Args:
            ref_gray:  Reference grayscale image (uint8 or float32, H×W).
            test_gray: Test grayscale image (uint8 or float32, H×W).
            config:    Config instance; all iteration parameters are read from
                       ``config.registration``.

        Returns:
            RegistrationResult with the found transform and quality score.
            ``status`` is ``Aligned`` when quality ≥
            ``config.registration.min_alignment_quality``, otherwise
            ``Failed``.

        Raises:
            NonConvergenceError: If ECC did not converge within the configured
                                 iteration limit (``ecc_max_iterations``).
        """
        # ECC requires float32 inputs
        ref_f = ref_gray.astype(np.float32)
        test_f = test_gray.astype(np.float32)

        # Initial identity warp matrix
        if self._motion_type == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = _termination_criteria(config)

        try:
            _, transform = cv2.findTransformECC(
                ref_f,
                test_f,
                warp_matrix,
                self._motion_type,
                criteria,
            )
        except cv2.error as exc:
            raise NonConvergenceError(
                "Image alignment did not converge. The images may be too different "
                "or lack enough texture for this method.",
                detail=f"cv2.findTransformECC ({self._method_name}): {exc}",
            ) from exc

        quality = _compute_quality(ref_gray, test_gray, transform, self._motion_type)
        logger.debug(
            "%s: quality=%.3f (min=%.3f).",
            self._method_name,
            quality,
            config.registration.min_alignment_quality,
        )

        status = (
            RegistrationStatus.Aligned
            if quality >= config.registration.min_alignment_quality
            else RegistrationStatus.Failed
        )

        return RegistrationResult(
            transform=transform,
            quality=quality,
            method=self._method_name,
            status=status,
            transform_type=self._transform_type,
        )


# ---------------------------------------------------------------------------
# Convenience singletons — used by the cascade and tests
# ---------------------------------------------------------------------------

ECC_EUCLIDEAN = ECCRegistrar(cv2.MOTION_EUCLIDEAN)
ECC_AFFINE = ECCRegistrar(cv2.MOTION_AFFINE)
