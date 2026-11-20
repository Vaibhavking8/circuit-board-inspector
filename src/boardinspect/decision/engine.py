"""
decision/engine.py — per-part presence/absence decision engine.

The engine maps per-signal scores and calibration statistics to PartResult
(Present / Missing / Uncertain) and aggregates part results to ImageStatus.
All logic reads from Config; no magic numbers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.models import (
    AlignmentInfo,
    CalibrationResult,
    ImageResult,
    ImageStatus,
    PartResult,
    PartStatus,
    TransformType,
)
from boardinspect.registration.interface import RegistrationStatus

logger = logging.getLogger(__name__)


@dataclass
class BorderInfo:
    """Geometric information about a part's proximity to the image border.

    ``near_border`` is True when the part's aligned box falls within
    config.decision.border_fraction of any image edge or is partly outside.
    ``reason`` is a short explanation when near_border is True.
    """
    near_border: bool
    reason: str = ""


def _z_margin(
    score: float,
    cal: CalibrationResult,
    signal_name: str,
) -> float:
    """Compute a normalised margin for one signal from calibration statistics.

    The margin is positive when the score is on the 'present' side of the
    threshold, and negative when on the 'absent' side.

    If calibration statistics are unavailable for the signal, falls back to
    (score - threshold) / 1.0.
    """
    stats = cal.signal_stats.get(signal_name)
    if stats is None:
        return score - cal.threshold

    present_stats = stats.get("present")
    absent_stats = stats.get("absent")

    if present_stats is None or absent_stats is None:
        return score - cal.threshold

    # Normalised distance: how far is the score from the threshold relative
    # to the gap between the two distribution means?
    gap = present_stats.mean - absent_stats.mean
    if abs(gap) < 1e-9:
        # Distributions fully overlap — calibration unreliable
        return score - cal.threshold

    return (score - cal.threshold) / abs(gap)


def check_border(
    inspection_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    config: Config = DEFAULT_CONFIG,
) -> BorderInfo:
    """Return BorderInfo for a part box in inspection-frame pixels.

    A part is 'near border' if any edge of its bounding box is within
    config.decision.border_fraction of the corresponding image edge, or if
    the box extends outside the image.

    Args:
        inspection_box: (x, y, w, h) in inspection-image pixels.
        image_width:    Inspection image width.
        image_height:   Inspection image height.
        config:         Config instance.

    Returns:
        BorderInfo(near_border=True/False, reason=...).
    """
    bx, by, bw, bh = inspection_box
    bx2, by2 = bx + bw, by + bh

    frac = config.decision.border_fraction
    margin_x = int(frac * image_width)
    margin_y = int(frac * image_height)

    reasons = []
    if bx < 0 or by < 0 or bx2 > image_width or by2 > image_height:
        reasons.append("box partially outside image")
    else:
        if bx < margin_x:
            reasons.append("near left edge")
        if bx2 > image_width - margin_x:
            reasons.append("near right edge")
        if by < margin_y:
            reasons.append("near top edge")
        if by2 > image_height - margin_y:
            reasons.append("near bottom edge")

    if reasons:
        return BorderInfo(near_border=True, reason="part near image border: " + ", ".join(reasons))
    return BorderInfo(near_border=False)


class DecisionEngine:
    """Map per-signal scores and calibration to per-part presence decisions."""

    def decide(
        self,
        part_id: str,
        signals: dict[str, float],
        calibration_result: CalibrationResult,
        border_info: BorderInfo,
        sensitivity: float,
        inspection_box: tuple[int, int, int, int],
        config: Config = DEFAULT_CONFIG,
    ) -> PartResult:
        """Decide presence/absence for one part.

        Args:
            part_id:            Stable part identifier.
            signals:            Map of signal name → raw score.
            calibration_result: Calibration data for this part.
            border_info:        Whether the aligned box is near/outside the border.
            sensitivity:        User-facing sensitivity [0, 1]; 0 = lenient, 1 = strict.
            inspection_box:     (x, y, w, h) in inspection-frame pixels.
            config:             Config instance.

        Returns:
            PartResult with status, margin, signals, threshold, and reason.
        """
        # 1. Shift threshold by sensitivity
        threshold = _shift_threshold(calibration_result.threshold, sensitivity, config)

        # 2. Compute weighted z-score margin
        comp = config.comparison
        total_weight = 0.0
        weighted_margin = 0.0
        for sig_name, score in signals.items():
            weight = comp.signal_weights.get(sig_name, 1.0)
            margin = _z_margin(score, calibration_result, sig_name)
            weighted_margin += weight * margin
            total_weight += weight

        if total_weight > 0:
            combined_margin = weighted_margin / total_weight
        else:
            combined_margin = 0.0

        logger.debug(
            "Decision part=%s: signals=%s, threshold=%.3f, combined_margin=%.3f.",
            part_id, signals, threshold, combined_margin,
        )

        # 3. Border check → always Uncertain
        if border_info.near_border:
            return PartResult(
                part_id=part_id,
                status=PartStatus.Uncertain,
                signals=signals,
                threshold=threshold,
                margin=combined_margin,
                inspection_box=inspection_box,
                reason=border_info.reason,
            )

        # 4. Uncertainty band
        uncertain_margin = config.decision.uncertain_margin
        if abs(combined_margin) < uncertain_margin:
            return PartResult(
                part_id=part_id,
                status=PartStatus.Uncertain,
                signals=signals,
                threshold=threshold,
                margin=combined_margin,
                inspection_box=inspection_box,
                reason="score within uncertainty band",
            )

        # 5. Final Present / Missing
        status = PartStatus.Present if combined_margin > 0 else PartStatus.Missing
        return PartResult(
            part_id=part_id,
            status=status,
            signals=signals,
            threshold=threshold,
            margin=combined_margin,
            inspection_box=inspection_box,
            reason=None,
        )


def _shift_threshold(
    base_threshold: float,
    sensitivity: float,
    config: Config,
) -> float:
    """Shift the base threshold by sensitivity.

    sensitivity=0.0 (lenient) → raise threshold (harder to flag Missing).
    sensitivity=1.0 (strict)  → lower threshold (easier to flag Missing).
    sensitivity=0.5           → use base_threshold unchanged.

    The shift is proportional to config.decision.uncertain_margin so that at
    the extremes the threshold moves by exactly one uncertain_margin.
    """
    dec = config.decision
    shift_range = dec.uncertain_margin  # max shift = ±uncertain_margin
    # Map [0, 1] → [-shift_range, +shift_range]; 0.5 maps to 0
    shift = (0.5 - sensitivity) * 2.0 * shift_range
    return base_threshold + shift


def aggregate_image_status(
    part_results: list[PartResult],
    alignment_status: RegistrationStatus,
) -> ImageStatus:
    """Aggregate part-level results to an overall image status.

    Rules (applied in order):
    1. Alignment failed → CouldNotVerify (never OK — a failure must not be OK).
    2. Any part Missing → MissingParts.
    3. Any part Uncertain and no part Missing → CouldNotVerify.
    4. All parts Present and alignment OK → OK.

    Args:
        part_results:     List of PartResult for all annotated parts.
        alignment_status: Outcome of the registration step.

    Returns:
        ImageStatus enum value.
    """
    # Rule 1: alignment failure always → CouldNotVerify
    if alignment_status == RegistrationStatus.Failed:
        return ImageStatus.CouldNotVerify

    has_missing = any(r.status == PartStatus.Missing for r in part_results)
    has_uncertain = any(r.status == PartStatus.Uncertain for r in part_results)

    if has_missing:
        return ImageStatus.MissingParts

    if has_uncertain:
        return ImageStatus.CouldNotVerify

    return ImageStatus.OK


# Singleton
DECISION_ENGINE = DecisionEngine()
