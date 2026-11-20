"""
pipeline.py — orchestrates one PCB inspection from raw bytes to ImageResult.

Steps:
  1. Load and validate the inspection image.
  2. Aspect-ratio check against the reference.
  3. Preprocess both images to registration resolution (grayscale).
  4. Register (align) via the cascade.  Failed → CouldNotVerify immediately.
  5. Scale the registration transform to comparison resolution.
  6. Warp the inspection image to the reference frame at comparison resolution.
  7. For each part: extract ref crop, compute search window, run signals.
  8. Decide per-part status via DecisionEngine.
  9. Aggregate to ImageStatus.
 10. Record runtime.

May import imaging, registration, comparison, decision, config, models,
coordinates.  Must NOT import api, storage, or the web framework.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import Callable

import cv2
import numpy as np

from boardinspect.comparison.edge import EDGE_GRADIENT_SIGNAL
from boardinspect.comparison.template import TEMPLATE_MATCH_SIGNAL
from boardinspect.comparison.window import compute_search_window
from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.coordinates import normalized_to_pixel
from boardinspect.decision.calibration import CALIBRATION_ENGINE
from boardinspect.decision.engine import (
    DECISION_ENGINE,
    BorderInfo,
    aggregate_image_status,
    check_border,
)
from boardinspect.errors import AlignmentError, AspectRatioError, ImageDecodeError
from boardinspect.imaging.load import ValidatedImage, validate_image
from boardinspect.imaging.preprocess import preprocess, to_grayscale
from boardinspect.models import (
    AlignmentInfo,
    CalibrationResult,
    ImageResult,
    ImageStatus,
    Part,
    PartResult,
    PartStatus,
    ReferenceSetup,
    SignalStats,
    TransformType,
)
from boardinspect.registration.cascade import REGISTRATION_CASCADE
from boardinspect.registration.interface import RegistrationResult, RegistrationStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal registry
# ---------------------------------------------------------------------------

_SIGNALS = {
    "template": TEMPLATE_MATCH_SIGNAL,
    "edge": EDGE_GRADIENT_SIGNAL,
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _aspect_ratio_ok(
    ref_w: int,
    ref_h: int,
    insp_w: int,
    insp_h: int,
    tolerance: float = 0.15,
) -> bool:
    """Return True when the two aspect ratios are within *tolerance* of each other.

    The tolerance is relative to the larger of the two ratios, so it scales
    naturally for both landscape and portrait images.
    """
    ref_ar = ref_w / ref_h if ref_h > 0 else 1.0
    insp_ar = insp_w / insp_h if insp_h > 0 else 1.0
    return abs(ref_ar - insp_ar) / max(ref_ar, insp_ar) < tolerance


def _scale_transform(
    transform: np.ndarray,
    sx: float,
    sy: float,
) -> np.ndarray:
    """Return a copy of *transform* scaled from one resolution to another.

    The registration transform was estimated at registration resolution.
    This scales it to comparison resolution by pre-multiplying with a
    scaling matrix S and post-multiplying with its inverse, following the
    standard change-of-coordinates formula: T' = S · T · S⁻¹.

    Works for both 2×3 affine and 3×3 homography matrices.
    """
    t = transform.copy().astype(np.float64)
    if t.shape == (3, 3):
        S = np.array([[sx, 0.0, 0.0],
                      [0.0, sy, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        S_inv = np.array([[1.0 / sx, 0.0, 0.0],
                          [0.0, 1.0 / sy, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        return (S @ t @ S_inv).astype(np.float32)
    else:
        # Affine 2×3: only the translation columns need scaling
        t[0, 2] *= sx
        t[1, 2] *= sy
        return t.astype(np.float32)


def _warp_image(
    image: np.ndarray,
    transform: np.ndarray,
    target_w: int,
    target_h: int,
    is_identity: bool,
) -> np.ndarray:
    """Warp *image* to the reference frame using *transform*.

    For the identity case the image is resized without warping. For affine
    (2×3) and homography (3×3) transforms the appropriate OpenCV warp is used.
    """
    if is_identity:
        h, w = image.shape[:2]
        if w != target_w or h != target_h:
            return cv2.resize(image, (target_w, target_h),
                              interpolation=cv2.INTER_AREA)
        return image

    if transform.shape == (3, 3):
        return cv2.warpPerspective(
            image, transform, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
        )
    # Affine 2×3
    return cv2.warpAffine(
        image, transform, (target_w, target_h),
        flags=cv2.INTER_LINEAR,
    )


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert an affine (2×3) or homography (3×3) transform matrix.

    Returns a float32 matrix of the same shape.
    """
    if transform.shape == (3, 3):
        return np.linalg.inv(transform.astype(np.float64)).astype(np.float32)
    # Extend 2×3 → 3×3, invert, take top two rows
    M = np.vstack([transform.astype(np.float64),
                   [0.0, 0.0, 1.0]])
    return np.linalg.inv(M)[:2].astype(np.float32)


def _transform_box(
    box_px: tuple[int, int, int, int],
    inv_transform: np.ndarray,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    """Map a pixel box through *inv_transform* (reference → inspection frame).

    Projects the four corners and returns the axis-aligned bounding box of the
    result, clamped to the inspection image dimensions.

    Args:
        box_px:        (x, y, w, h) in the aligned reference frame.
        inv_transform: Inverse transform (reference → inspection frame).
        image_w:       Inspection image width.
        image_h:       Inspection image height.

    Returns:
        (x, y, w, h) in the inspection image frame.
    """
    bx, by, bw, bh = box_px
    corners = np.array(
        [[bx,       by      ],
         [bx + bw,  by      ],
         [bx + bw,  by + bh ],
         [bx,       by + bh ]],
        dtype=np.float32,
    )

    # Homogeneous multiply
    ones = np.ones((4, 1), dtype=np.float32)
    h_corners = np.hstack([corners, ones])  # (4, 3)

    if inv_transform.shape == (3, 3):
        t = (inv_transform.astype(np.float32) @ h_corners.T).T  # (4, 3)
        pts = t[:, :2] / t[:, 2:3]
    else:
        pts = (inv_transform.astype(np.float32) @ h_corners.T).T  # (4, 2)

    xs = np.clip(pts[:, 0], 0, image_w - 1)
    ys = np.clip(pts[:, 1], 0, image_h - 1)

    ix = int(xs.min())
    iy = int(ys.min())
    iw = max(1, int(xs.max()) - ix)
    ih = max(1, int(ys.max()) - iy)
    return (ix, iy, iw, ih)


def _default_calibration(enabled_signals: list[str]) -> CalibrationResult:
    """Build a minimal fallback CalibrationResult when none is stored.

    Uses a midpoint threshold of 0.6 and marks the result as unreliable so
    the decision engine knows not to trust the threshold tightly.
    """
    return CalibrationResult(
        signal_stats={
            sig: {
                "present": SignalStats(mean=0.9, std=0.1),
                "absent":  SignalStats(mean=0.3, std=0.1),
            }
            for sig in enabled_signals
        },
        threshold=0.6,
        reliable=False,
        reason="No calibration data; using default threshold",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_one_with_ref(
    reference_setup: ReferenceSetup,
    reference_image: np.ndarray,
    inspection_bytes: bytes,
    sensitivity: float = 0.5,
    config: Config = DEFAULT_CONFIG,
) -> ImageResult:
    """Run one PCB inspection with the reference image provided.

    This is the main pipeline entry point. The API service layer loads the
    reference image from storage and passes it here so the pipeline has no
    disk I/O responsibility.

    Args:
        reference_setup:  Annotated and calibrated reference setup.
        reference_image:  BGR reference image (uint8, H×W×3).
        inspection_bytes: Raw bytes of the inspection image.
        sensitivity:      User sensitivity [0, 1].  0 = lenient, 1 = strict.
        config:           Config instance.

    Returns:
        ImageResult.  The status is never OK when any failure path is taken;
        a failure always produces CouldNotVerify (R9.1, R9.2).

    Raises:
        ImageDecodeError, ImageTooLargeError, ImageTooSmallError — propagated
        unchanged so the caller can distinguish validation failures from
        alignment failures.
    """
    t_start = time.monotonic()
    messages: list[str] = []
    ref_w = reference_setup.original_width
    ref_h = reference_setup.original_height

    # ------------------------------------------------------------------
    # Step 1: Decode the inspection image to get dimensions, then run the
    # aspect-ratio check before the full size/megapixel validation.  This
    # lets us return CouldNotVerify cleanly when the image is obviously the
    # wrong board, without a confusing "image too small" error that would
    # be raised for a portrait-vs-landscape mismatch test image.
    # ------------------------------------------------------------------
    from boardinspect.imaging.load import load_image as _load_raw

    # Minimal decode to get dimensions (skips size/megapixel limits)
    _raw_arr = np.frombuffer(inspection_bytes, dtype=np.uint8)
    _decoded = cv2.imdecode(_raw_arr, cv2.IMREAD_COLOR)
    if _decoded is None:
        # Will be reported properly by validate_image below
        insp_w, insp_h = 0, 0
    else:
        insp_h, insp_w = _decoded.shape[:2]

    # ------------------------------------------------------------------
    # Step 2: Aspect-ratio check (before full validation)
    # ------------------------------------------------------------------
    if insp_w > 0 and not _aspect_ratio_ok(ref_w, ref_h, insp_w, insp_h):
        msg = (
            f"Inspection image aspect ratio ({insp_w}×{insp_h}) differs "
            f"significantly from the reference ({ref_w}×{ref_h}). "
            "The images may be of different boards or taken in different "
            "orientations."
        )
        messages.append(msg)
        logger.warning(msg)
        return ImageResult(
            status=ImageStatus.CouldNotVerify,
            alignment=AlignmentInfo(
                strategy="aspect_ratio_check",
                transform_type=TransformType.Identity,
                quality=0.0,
            ),
            part_results=[],
            runtime_ms=(time.monotonic() - t_start) * 1000.0,
            messages=messages,
        )

    # ------------------------------------------------------------------
    # Step 2b: Full validation (size, megapixel limits, proper decode)
    # Only reached when the aspect ratio is acceptable.
    # ------------------------------------------------------------------
    validated: ValidatedImage = validate_image(
        inspection_bytes, "<inspection>", config
    )
    insp_bgr = validated.array

    # ------------------------------------------------------------------
    # Step 3: Preprocess both images to registration resolution
    # ------------------------------------------------------------------
    ref_reg = preprocess(reference_image, config, stage="registration")
    insp_reg = preprocess(insp_bgr, config, stage="registration")

    reg_h, reg_w = ref_reg.shape[:2]

    # ------------------------------------------------------------------
    # Step 4: Register (align) via the cascade
    # ------------------------------------------------------------------
    reg_result: RegistrationResult = REGISTRATION_CASCADE.register(
        ref_reg, insp_reg, config
    )

    if reg_result.status == RegistrationStatus.Failed:
        msg = "Could not align the inspection image to the reference."
        messages.append(msg)
        logger.warning(msg)
        return ImageResult(
            status=ImageStatus.CouldNotVerify,
            alignment=AlignmentInfo(
                strategy=reg_result.method,
                transform_type=reg_result.transform_type,
                quality=reg_result.quality,
            ),
            part_results=[],
            runtime_ms=(time.monotonic() - t_start) * 1000.0,
            messages=messages,
        )

    alignment_info = AlignmentInfo(
        strategy=reg_result.method,
        transform_type=reg_result.transform_type,
        quality=reg_result.quality,
    )
    is_identity = reg_result.status == RegistrationStatus.Identity

    # ------------------------------------------------------------------
    # Step 5: Scale the transform to comparison resolution and warp
    # ------------------------------------------------------------------
    ref_comp = preprocess(reference_image, config, stage="comparison")
    insp_comp = preprocess(insp_bgr, config, stage="comparison")

    comp_h, comp_w = ref_comp.shape[:2]

    # Scale factors from registration resolution to comparison resolution
    sx = comp_w / reg_w if reg_w > 0 else 1.0
    sy = comp_h / reg_h if reg_h > 0 else 1.0

    if is_identity:
        comp_transform = np.eye(2, 3, dtype=np.float32)
    else:
        comp_transform = _scale_transform(reg_result.transform, sx, sy)

    warped_insp = _warp_image(insp_comp, comp_transform, comp_w, comp_h, is_identity)

    # ------------------------------------------------------------------
    # Step 6–8: Per-part comparison and decision
    # ------------------------------------------------------------------
    calibration: dict[str, CalibrationResult] = reference_setup.calibration or {}
    parts = reference_setup.parts
    enabled_signals = config.comparison.signals_enabled

    # Pre-compute all pixel boxes at comparison resolution
    all_boxes_px = [
        normalized_to_pixel(
            p.box, comp_w, comp_h,
            min_size=config.part_limits.min_box_size_px,
        )
        for p in parts
    ]

    # Inverse transform for projecting reference-frame boxes back to the
    # inspection image frame (used for overlay geometry only)
    if is_identity:
        inv_transform = np.eye(2, 3, dtype=np.float32)
    else:
        inv_transform = _invert_transform(comp_transform)

    part_results: list[PartResult] = []

    for idx, (part, box_px) in enumerate(zip(parts, all_boxes_px)):
        bx, by, bw, bh = box_px

        # Extract reference crop at comparison resolution
        ref_crop = ref_comp[by: by + bh, bx: bx + bw]
        if ref_crop.size == 0:
            logger.warning(
                "Part '%s': empty crop at comparison resolution; marking Uncertain.",
                part.name,
            )
            part_results.append(PartResult(
                part_id=part.id,
                status=PartStatus.Uncertain,
                signals={},
                threshold=0.5,
                margin=0.0,
                inspection_box=(0, 0, 1, 1),
                reason="Empty reference crop at comparison resolution",
            ))
            continue

        # Search window in the warped inspection image
        other_boxes = [b for i, b in enumerate(all_boxes_px) if i != idx]
        window = compute_search_window(
            box_px, other_boxes, comp_w, comp_h, config
        )
        search_region = warped_insp[
            window.y: window.y + window.h,
            window.x: window.x + window.w,
        ]

        # Run each enabled signal
        signals: dict[str, float] = {}
        for sig_name in enabled_signals:
            signal = _SIGNALS.get(sig_name)
            if signal is None:
                logger.warning("Unknown signal '%s'; skipping.", sig_name)
                continue
            sig_result = signal.compare(ref_crop, search_region, config)
            signals[sig_name] = sig_result.value
            logger.debug(
                "Part '%s' signal '%s': %.3f.",
                part.name, sig_name, sig_result.value,
            )

        # Map the reference-frame box to the inspection image frame
        if is_identity:
            inspection_box = box_px
        else:
            inspection_box = _transform_box(
                box_px, inv_transform, comp_w, comp_h
            )

        border_info = check_border(inspection_box, comp_w, comp_h, config)

        # Calibration: use stored result or fall back to defaults
        cal_result = calibration.get(part.id)
        if cal_result is None:
            cal_result = _default_calibration(enabled_signals)
            messages.append(
                f"Part '{part.name}': no calibration data, using default threshold."
            )

        pr = DECISION_ENGINE.decide(
            part_id=part.id,
            signals=signals,
            calibration_result=cal_result,
            border_info=border_info,
            sensitivity=sensitivity,
            inspection_box=inspection_box,
            config=config,
        )
        part_results.append(pr)

    # ------------------------------------------------------------------
    # Step 9: Aggregate image status
    # ------------------------------------------------------------------
    image_status = aggregate_image_status(part_results, reg_result.status)

    runtime_ms = (time.monotonic() - t_start) * 1000.0
    logger.debug(
        "inspect_one_with_ref: status=%s, %d parts, %.1f ms.",
        image_status, len(part_results), runtime_ms,
    )

    return ImageResult(
        status=image_status,
        alignment=alignment_info,
        part_results=part_results,
        runtime_ms=runtime_ms,
        messages=messages,
    )


def calibrate_reference(
    reference_image: np.ndarray,
    reference_setup: ReferenceSetup,
    config: Config = DEFAULT_CONFIG,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ReferenceSetup:
    """Calibrate all parts and return an updated ReferenceSetup.

    Runs the CalibrationEngine against the reference image to produce
    per-part score distributions and decision thresholds, then returns
    a new ReferenceSetup with the calibration field populated.

    Args:
        reference_image:   BGR reference image (uint8, H×W×3).
        reference_setup:   Setup with annotated parts (calibration may be None).
        config:            Config instance.
        progress_callback: Optional callback(part_index, total_parts).

    Returns:
        New ReferenceSetup with calibration populated.

    Raises:
        CalibrationError: If a part crop cannot be extracted (image too small,
                          etc.).
    """
    calibration = CALIBRATION_ENGINE.calibrate(
        reference_image,
        reference_setup.parts,
        config,
        progress_callback,
    )
    return dataclasses.replace(reference_setup, calibration=calibration)
