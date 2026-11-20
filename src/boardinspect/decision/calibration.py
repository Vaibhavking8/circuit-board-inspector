"""
decision/calibration.py — per-part synthetic-augmentation calibration.

For each part, the calibration engine generates 'present with variation' and
'absent' (inpainted) augmented samples from the reference image, scores them
with the enabled comparison signals, and places a decision threshold between
the two score distributions.

Deterministic: numpy.random.default_rng seeded from config.random_seed + part_index.
No web-framework, storage, or imaging layer imports (pure CV).
"""
from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from boardinspect.comparison.edge import EDGE_GRADIENT_SIGNAL
from boardinspect.comparison.template import TEMPLATE_MATCH_SIGNAL
from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.coordinates import normalized_to_pixel
from boardinspect.errors import CalibrationError
from boardinspect.models import (
    CalibrationResult,
    Part,
    SignalStats,
)

logger = logging.getLogger(__name__)

_SIGNALS = {
    "template": TEMPLATE_MATCH_SIGNAL,
    "edge": EDGE_GRADIENT_SIGNAL,
}

# Inpainting radius for absent sample generation
_INPAINT_RADIUS = 5


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment(
    image: np.ndarray,
    rng: np.random.Generator,
    config: Config,
) -> np.ndarray:
    """Apply random photometric and geometric augmentation to a grayscale crop.

    All variation ranges come from config.calibration. The augmented image
    is returned as uint8.
    """
    cal = config.calibration
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    # 1. Brightness variation
    brightness_shift = rng.uniform(-cal.brightness, cal.brightness) * 255.0
    img = np.clip(img + brightness_shift, 0, 255)

    # 2. Contrast variation
    contrast_factor = 1.0 + rng.uniform(-cal.contrast, cal.contrast)
    img = np.clip((img - 128) * contrast_factor + 128, 0, 255)

    # 3. Gaussian noise
    noise = rng.normal(0, cal.noise_std, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    img = img.astype(np.uint8)

    # 4. Small rotation
    angle = rng.uniform(-cal.rotation_degrees, cal.rotation_degrees)
    if abs(angle) > 0.1:
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # 5. Small translation / shift
    dx = int(rng.integers(-cal.shift_px, cal.shift_px + 1))
    dy = int(rng.integers(-cal.shift_px, cal.shift_px + 1))
    if dx != 0 or dy != 0:
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, T, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    # 6. Optional mild blur
    if rng.random() < 0.3:
        k = cal.blur_kernel if cal.blur_kernel % 2 == 1 else cal.blur_kernel + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def _make_absent_image(
    reference: np.ndarray,
    box_px: tuple[int, int, int, int],
) -> np.ndarray:
    """Generate an 'absent' version of the reference by inpainting the part region.

    Args:
        reference: Grayscale reference image (uint8, H×W).
        box_px:    (x, y, w, h) of the part in reference pixels.

    Returns:
        Grayscale image with the part region filled with surrounding texture.
    """
    bx, by, bw, bh = box_px
    mask = np.zeros(reference.shape[:2], dtype=np.uint8)
    # Expand mask slightly (1px border) to erase the part fully
    mx1 = max(0, bx - 1)
    my1 = max(0, by - 1)
    mx2 = min(reference.shape[1], bx + bw + 1)
    my2 = min(reference.shape[0], by + bh + 1)
    mask[my1:my2, mx1:mx2] = 255
    return cv2.inpaint(reference, mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA)


def _score_crop(
    ref_crop: np.ndarray,
    test_crop: np.ndarray,
    signal_name: str,
    config: Config,
) -> float:
    """Run one signal comparing ref_crop against a padded region from test_crop.

    For calibration, the 'search region' is the test_crop itself (no additional
    padding — the crop is already the right size).
    """
    signal = _SIGNALS.get(signal_name)
    if signal is None:
        logger.warning("Unknown signal '%s'; skipping.", signal_name)
        return 0.0
    result = signal.compare(ref_crop, test_crop, config)
    return result.value


# ---------------------------------------------------------------------------
# CalibrationEngine
# ---------------------------------------------------------------------------

class CalibrationEngine:
    """Calibrate per-part thresholds using synthetic augmentation."""

    def calibrate(
        self,
        reference_image: np.ndarray,
        parts: list[Part],
        config: Config = DEFAULT_CONFIG,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, CalibrationResult]:
        """Calibrate all parts against the reference image.

        Args:
            reference_image:   Grayscale reference image (uint8, H×W).
                               If colour, it is converted to grayscale internally.
            parts:             List of Part objects to calibrate.
            config:            Config instance.
            progress_callback: Optional callback(part_index, total_parts) called
                               after each part is calibrated.

        Returns:
            Map of part_id → CalibrationResult.

        Raises:
            CalibrationError: If calibration fails for a critical reason (e.g.
                               the reference image is too small to extract any part).
        """
        # Convert to grayscale if needed
        if reference_image.ndim == 3:
            gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_ref = reference_image

        h, w = gray_ref.shape[:2]
        cal_cfg = config.calibration
        n_samples = cal_cfg.sample_count
        total = len(parts)

        results: dict[str, CalibrationResult] = {}

        for idx, part in enumerate(parts):
            rng = np.random.default_rng(config.random_seed + idx)

            # Convert normalised box to pixels
            box_px = normalized_to_pixel(part.box, w, h, min_size=config.part_limits.min_box_size_px)
            bx, by, bw, bh = box_px

            if bw <= 0 or bh <= 0:
                raise CalibrationError(
                    f"Part '{part.name}' has a zero-size pixel box; cannot calibrate.",
                    detail=f"box_px={box_px}, image={w}x{h}",
                )

            ref_crop = gray_ref[by:by + bh, bx:bx + bw].copy()

            # Build absent reference via inpainting
            absent_ref = _make_absent_image(gray_ref, box_px)
            absent_crop = absent_ref[by:by + bh, bx:bx + bw].copy()

            enabled_signals = config.comparison.signals_enabled

            # Collect scores for each signal under each condition
            present_scores: dict[str, list[float]] = {s: [] for s in enabled_signals}
            absent_scores: dict[str, list[float]] = {s: [] for s in enabled_signals}

            for _ in range(n_samples):
                aug_present = _augment(ref_crop, rng, config)
                aug_absent = _augment(absent_crop, rng, config)

                for sig_name in enabled_signals:
                    # Build a padded search region so matchTemplate has room to slide.
                    # Padding is a fraction of the box size, minimum 5px.
                    pad = max(5, min(bw, bh) // 4)
                    sx = max(0, bx - pad)
                    sy = max(0, by - pad)
                    ex = min(w, bx + bw + pad)
                    ey = min(h, by + bh + pad)

                    # Present: insert augmented present crop into reference context
                    search_present = gray_ref[sy:ey, sx:ex].copy()
                    # Coordinates of part within the search window
                    part_in_win_x = bx - sx
                    part_in_win_y = by - sy
                    search_present[
                        part_in_win_y: part_in_win_y + bh,
                        part_in_win_x: part_in_win_x + bw,
                    ] = aug_present

                    # Absent: insert augmented absent crop into absent-reference context
                    search_absent = absent_ref[sy:ey, sx:ex].copy()
                    search_absent[
                        part_in_win_y: part_in_win_y + bh,
                        part_in_win_x: part_in_win_x + bw,
                    ] = aug_absent

                    p_score = _score_crop(ref_crop, search_present, sig_name, config)
                    a_score = _score_crop(ref_crop, search_absent, sig_name, config)
                    present_scores[sig_name].append(p_score)
                    absent_scores[sig_name].append(a_score)

            # Build signal statistics and per-signal midpoint thresholds
            signal_stats: dict[str, dict[str, SignalStats]] = {}
            thresholds: list[float] = []
            any_unreliable = False
            unreliable_reasons: list[str] = []

            for sig_name in enabled_signals:
                p_arr = np.array(present_scores[sig_name])
                a_arr = np.array(absent_scores[sig_name])

                p_mean, p_std = float(p_arr.mean()), float(p_arr.std())
                a_mean, a_std = float(a_arr.mean()), float(a_arr.std())

                signal_stats[sig_name] = {
                    "present": SignalStats(mean=p_mean, std=p_std),
                    "absent": SignalStats(mean=a_mean, std=a_std),
                }

                midpoint = (p_mean + a_mean) / 2.0
                thresholds.append(midpoint)

                # Reliability: present distribution must clearly exceed absent
                if p_mean <= a_mean + cal_cfg.threshold_margin:
                    any_unreliable = True
                    unreliable_reasons.append(
                        f"{sig_name}: present mean ({p_mean:.3f}) not clearly above "
                        f"absent mean ({a_mean:.3f}) + margin ({cal_cfg.threshold_margin})"
                    )

            # Combined threshold: average of per-signal midpoints
            threshold = float(np.mean(thresholds)) if thresholds else 0.5
            reliable = not any_unreliable
            reason = "; ".join(unreliable_reasons) if unreliable_reasons else ""

            results[part.id] = CalibrationResult(
                signal_stats=signal_stats,
                threshold=threshold,
                reliable=reliable,
                reason=reason,
            )

            logger.debug(
                "Calibrated part '%s' (id=%s): threshold=%.3f, reliable=%s.",
                part.name, part.id, threshold, reliable,
            )

            if progress_callback is not None:
                progress_callback(idx + 1, total)

        return results


# Singleton
CALIBRATION_ENGINE = CalibrationEngine()
