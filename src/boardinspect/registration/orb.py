"""
registration/orb.py — ORB feature matching + RANSAC homography fallback.

Used when ECC strategies fail (e.g. large rotation, strong lighting change).
Quality is the RANSAC inlier ratio. If too few features or matches are found,
LowTextureError is raised so the cascade can skip this strategy gracefully.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.errors import LowTextureError, RepetitivePatternError
from boardinspect.models import TransformType
from boardinspect.registration.interface import RegistrationResult, RegistrationStatus

logger = logging.getLogger(__name__)


def _homography_to_affine(H: np.ndarray) -> np.ndarray | None:
    """Extract a 2×3 affine approximation from a 3×3 homography.

    If the perspective warp components (H[2, 0], H[2, 1]) are small relative
    to the affine components, the perspective component is negligible and the
    2×3 affine is a valid approximation. Otherwise return None to signal that
    the full homography should be kept as 3×3.

    Returns:
        2×3 float32 affine matrix, or None if perspective component is significant.
    """
    perspective_magnitude = np.sqrt(H[2, 0] ** 2 + H[2, 1] ** 2)
    if perspective_magnitude < 1e-4:
        return H[:2, :].astype(np.float32)
    return None


class ORBRegistrar:
    """Align images using ORB keypoints, BFMatcher (Hamming), and RANSAC homography.

    Args:
        config: Config instance (used at construction for feature count).
    """

    def register(
        self,
        ref_gray: np.ndarray,
        test_gray: np.ndarray,
        config: Config = DEFAULT_CONFIG,
    ) -> RegistrationResult:
        """Align *test_gray* to *ref_gray* using ORB + RANSAC.

        Args:
            ref_gray:  Reference grayscale image (uint8, H×W).
            test_gray: Test grayscale image (uint8, H×W).
            config:    Config instance with registration parameters.

        Returns:
            RegistrationResult. Status is Aligned if inlier ratio is above minimum.

        Raises:
            LowTextureError: Too few ORB features or matches found.
            RepetitivePatternError: RANSAC failed to find a consistent homography
                                    despite enough matches (repetitive texture).
        """
        reg = config.registration

        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=reg.orb_max_features)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        kp_test, des_test = orb.detectAndCompute(test_gray, None)

        n_ref = len(kp_ref) if kp_ref else 0
        n_test = len(kp_test) if kp_test else 0
        logger.debug("ORB: %d ref keypoints, %d test keypoints.", n_ref, n_test)

        if n_ref < reg.orb_min_matches or n_test < reg.orb_min_matches:
            raise LowTextureError(
                "Not enough features found in the images for feature-based alignment. "
                "The board area may be too uniform.",
                detail=f"ORB: ref={n_ref}, test={n_test} keypoints "
                       f"(min={reg.orb_min_matches})",
            )

        if des_ref is None or des_test is None:
            raise LowTextureError(
                "Feature descriptors could not be computed.",
                detail="ORB detectAndCompute returned None descriptors",
            )

        # Match with BFMatcher (Hamming for ORB binary descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_ref, des_test)
        matches = sorted(matches, key=lambda m: m.distance)

        logger.debug("ORB: %d matches after cross-check.", len(matches))

        if len(matches) < reg.orb_min_matches:
            raise LowTextureError(
                "Not enough feature matches for alignment. "
                "The images may be too different or lack texture.",
                detail=f"ORB: only {len(matches)} matches (min={reg.orb_min_matches})",
            )

        # Extract matched point coordinates
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in matches])
        pts_test = np.float32([kp_test[m.trainIdx].pt for m in matches])

        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(
            pts_test, pts_ref,
            cv2.RANSAC,
            reg.orb_ransac_threshold,
        )

        if H is None or mask is None:
            raise RepetitivePatternError(
                "Could not estimate a reliable alignment transform. "
                "The board texture may be too repetitive.",
                detail=f"cv2.findHomography returned None with {len(matches)} matches",
            )

        inlier_count = int(mask.sum())
        inlier_ratio = inlier_count / len(matches)
        logger.debug(
            "ORB RANSAC: %d/%d inliers (ratio=%.3f).",
            inlier_count, len(matches), inlier_ratio,
        )

        # Quality is the inlier ratio, clamped to [0, 1]
        quality = float(min(1.0, max(0.0, inlier_ratio)))

        # Try to reduce to affine if perspective component is negligible
        affine = _homography_to_affine(H)
        if affine is not None:
            transform = affine
            transform_type = TransformType.Affine
            logger.debug("ORB: perspective component negligible; using affine approximation.")
        else:
            transform = H.astype(np.float32)
            transform_type = TransformType.Homography

        status = (
            RegistrationStatus.Aligned
            if quality >= config.registration.min_alignment_quality
            else RegistrationStatus.Failed
        )

        return RegistrationResult(
            transform=transform,
            quality=quality,
            method="orb_homography",
            status=status,
            transform_type=transform_type,
        )


# Singleton for use by the cascade
ORB_REGISTRAR = ORBRegistrar()
