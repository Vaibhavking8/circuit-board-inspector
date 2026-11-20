"""
decision/sensitivity.py — sensitivity control for the decision engine.

A single user-facing sensitivity value (0.0 = lenient, 1.0 = strict)
shifts per-part thresholds monotonically. Higher sensitivity means the
threshold is lowered, so more parts are flagged Missing.

re_evaluate() re-decides all parts from stored signal values without
re-running alignment or comparison. Used by the API's re-evaluate route.
"""
from __future__ import annotations

import logging

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.decision.engine import (
    DECISION_ENGINE,
    BorderInfo,
    aggregate_image_status,
    check_border,
    _shift_threshold,
)
from boardinspect.models import (
    CalibrationResult,
    ImageStatus,
    PartResult,
    PartStatus,
    ReferenceSetup,
)
from boardinspect.registration.interface import RegistrationStatus

logger = logging.getLogger(__name__)


def shift_thresholds(
    calibration_results: dict[str, CalibrationResult],
    sensitivity: float,
    config: Config = DEFAULT_CONFIG,
) -> dict[str, float]:
    """Compute shifted thresholds for all parts at the given sensitivity.

    Args:
        calibration_results: Map of part_id → CalibrationResult.
        sensitivity:         Float in [0, 1]. 0=lenient, 1=strict, 0.5=default.
        config:              Config instance.

    Returns:
        Map of part_id → shifted threshold float.
    """
    result: dict[str, float] = {}
    for part_id, cal in calibration_results.items():
        result[part_id] = _shift_threshold(cal.threshold, sensitivity, config)
    return result


def re_evaluate(
    stored_part_results: list[PartResult],
    calibration_results: dict[str, CalibrationResult],
    reference_setup: ReferenceSetup,
    inspection_image_width: int,
    inspection_image_height: int,
    alignment_status: RegistrationStatus,
    sensitivity: float,
    config: Config = DEFAULT_CONFIG,
) -> tuple[list[PartResult], ImageStatus]:
    """Re-decide all parts from stored signal values at a new sensitivity.

    Does NOT re-run alignment or comparison — only re-applies the decision
    engine with shifted thresholds. Used by the re-evaluate API route.

    Args:
        stored_part_results:     PartResult list from the original inspection.
        calibration_results:     Map of part_id → CalibrationResult (from setup).
        reference_setup:         ReferenceSetup (for part metadata).
        inspection_image_width:  Width of the inspection image (for border check).
        inspection_image_height: Height of the inspection image (for border check).
        alignment_status:        Registration outcome from the original inspection.
        sensitivity:             New sensitivity value [0, 1].
        config:                  Config instance.

    Returns:
        Tuple of (new_part_results, new_image_status).
    """
    new_results: list[PartResult] = []

    for pr in stored_part_results:
        cal = calibration_results.get(pr.part_id)
        if cal is None:
            # No calibration for this part — keep the original result unchanged.
            new_results.append(pr)
            logger.warning(
                "re_evaluate: no calibration for part '%s'; keeping original.",
                pr.part_id,
            )
            continue

        # Re-compute border info from the stored inspection box.
        border_info = check_border(
            pr.inspection_box,
            inspection_image_width,
            inspection_image_height,
            config,
        )

        new_pr = DECISION_ENGINE.decide(
            part_id=pr.part_id,
            signals=pr.signals,
            calibration_result=cal,
            border_info=border_info,
            sensitivity=sensitivity,
            inspection_box=pr.inspection_box,
            config=config,
        )
        new_results.append(new_pr)

    new_status = aggregate_image_status(new_results, alignment_status)
    logger.debug(
        "re_evaluate: sensitivity=%.2f → %d results, status=%s.",
        sensitivity,
        len(new_results),
        new_status,
    )
    return new_results, new_status
