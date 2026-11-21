"""
api/services/inspection_service.py — thin adapter for inspection.

Delegates to pipeline.inspect_one_with_ref. No CV logic here.
"""
from __future__ import annotations

import logging

import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.models import ImageResult, ReferenceSetup
from boardinspect.pipeline import inspect_one_with_ref

logger = logging.getLogger(__name__)


def run_inspection(
    setup: ReferenceSetup,
    reference_image: np.ndarray,
    inspection_bytes: bytes,
    sensitivity: float = 0.5,
    config: Config = DEFAULT_CONFIG,
) -> ImageResult:
    """Run one inspection. Thin wrapper around pipeline.inspect_one_with_ref.

    Args:
        setup:             Calibrated ReferenceSetup.
        reference_image:   BGR reference image array.
        inspection_bytes:  Raw bytes of the inspection image.
        sensitivity:       Sensitivity [0, 1].
        config:            Config instance.

    Returns:
        ImageResult.
    """
    logger.debug("Starting inspection (sensitivity=%.2f)", sensitivity)
    return inspect_one_with_ref(setup, reference_image, inspection_bytes, sensitivity, config)
