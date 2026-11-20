"""
viz.py — server-side result overlays for export and evaluation reports.

Draws the same overlays as the frontend: red boxes + labels for missing parts,
a coloured status banner, and a side-by-side comparison view.

All functions are pure: they accept numpy arrays and return new arrays.
No web-framework, storage, or network imports.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.models import ImageResult, ImageStatus, PartStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour constants (BGR)
# ---------------------------------------------------------------------------
_GREEN = (60, 180, 60)
_RED   = (50, 50, 220)
_AMBER = (30, 165, 215)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

_STATUS_COLOUR: dict[ImageStatus, tuple[int, int, int]] = {
    ImageStatus.OK:             _GREEN,
    ImageStatus.MissingParts:   _RED,
    ImageStatus.CouldNotVerify: _AMBER,
}

_STATUS_LABEL: dict[ImageStatus, str] = {
    ImageStatus.OK:             "OK — No missing parts",
    ImageStatus.MissingParts:   "MISSING PARTS",
    ImageStatus.CouldNotVerify: "COULD NOT VERIFY",
}


def _put_text_with_bg(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    text_colour: tuple[int, int, int],
    bg_colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw text with a filled background rectangle (in-place).

    ``origin`` is the bottom-left corner of the text in OpenCV convention.
    The background rectangle is expanded by 2 px on each side so the text
    has a small margin and remains legible at all banner heights.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    # Background rectangle
    cv2.rectangle(
        image,
        (x - 2, y - th - 2),
        (x + tw + 2, y + baseline + 2),
        bg_colour,
        -1,
    )
    # Text on top
    cv2.putText(image, text, (x, y), font, font_scale, text_colour, thickness, cv2.LINE_AA)


def render_result_overlay(
    image: np.ndarray,
    image_result: ImageResult,
    config: Config = DEFAULT_CONFIG,
) -> np.ndarray:
    """Draw inspection overlays onto *image* and return the annotated copy.

    Overlays applied:
    - A coloured status banner across the top (green / red / amber).
    - For every part: a bounding box in the status colour of that part.
    - A name label above each box, with a matching filled background.

    Present parts → green box; Missing parts → red box; Uncertain → amber box.
    Text labels satisfy R8.5 (colour is never the only channel of information).

    ``inspection_box`` in each PartResult is already in pixel coordinates on
    the inspection image, so no additional coordinate conversion is needed here.

    Args:
        image:        BGR inspection image (uint8, H×W×3).
        image_result: ImageResult produced by the pipeline.
        config:       Config instance (kept for future configurable overlay
                      parameters; not used by the current implementation).

    Returns:
        New BGR image (same shape as *image*) with overlays drawn.
        The input array is never mutated.
    """
    out = image.copy()
    h, w = out.shape[:2]

    # --- Per-part boxes and labels ---
    for pr in image_result.part_results:
        bx, by, bw, bh = pr.inspection_box

        if pr.status == PartStatus.Missing:
            colour = _RED
        elif pr.status == PartStatus.Uncertain:
            colour = _AMBER
        else:
            colour = _GREEN

        # Box — thickness 2 is visible without obscuring small parts
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), colour, 2)

        # Label above the box; scale with box width so small parts still get text
        font_scale = max(0.3, min(0.6, bw / 100))
        _put_text_with_bg(
            out,
            pr.part_id,
            (max(0, bx), max(12, by - 4)),
            font_scale,
            _WHITE,
            colour,
        )

    # --- Status banner at the top ---
    banner_h = max(28, h // 20)
    banner_colour = _STATUS_COLOUR.get(image_result.status, _AMBER)
    cv2.rectangle(out, (0, 0), (w, banner_h), banner_colour, -1)

    banner_text = _STATUS_LABEL.get(image_result.status, str(image_result.status))
    missing_ids = [
        pr.part_id
        for pr in image_result.part_results
        if pr.status == PartStatus.Missing
    ]
    if missing_ids:
        banner_text += ": " + ", ".join(missing_ids)

    _put_text_with_bg(
        out,
        banner_text,
        (8, banner_h - 6),
        0.5,
        _WHITE,
        banner_colour,
        thickness=1,
    )

    logger.debug(
        "render_result_overlay: status=%s, %d parts drawn.",
        image_result.status,
        len(image_result.part_results),
    )
    return out


def render_side_by_side(
    ref_crop: np.ndarray,
    insp_crop: np.ndarray,
    part_id: str = "",
    separator_width: int = 4,
) -> np.ndarray:
    """Compose reference and inspection crops side by side.

    Both crops are scaled to the same height before concatenating.
    A thin dark separator is inserted between them, and a label bar is
    prepended at the top showing "Reference" and "Inspection" headings plus
    the optional ``part_id`` centred between them.

    Args:
        ref_crop:        BGR or grayscale reference crop.
        insp_crop:       BGR or grayscale inspection crop.
        part_id:         Optional label shown centred in the label bar.
        separator_width: Width of the vertical separator in pixels.

    Returns:
        BGR composite with shape (label_h + max(ref_h, insp_h),
        ref_w + separator_width + insp_w, 3).
    """
    # Ensure both panels are BGR
    if ref_crop.ndim == 2:
        ref_bgr = cv2.cvtColor(ref_crop, cv2.COLOR_GRAY2BGR)
    else:
        ref_bgr = ref_crop.copy()

    if insp_crop.ndim == 2:
        insp_bgr = cv2.cvtColor(insp_crop, cv2.COLOR_GRAY2BGR)
    else:
        insp_bgr = insp_crop.copy()

    # Match heights by scaling the shorter panel up
    target_h = max(ref_bgr.shape[0], insp_bgr.shape[0])

    def _scale_to_height(img: np.ndarray, target: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target:
            return img
        scale = target / h
        new_w = max(1, round(w * scale))
        return cv2.resize(img, (new_w, target), interpolation=cv2.INTER_AREA)

    ref_bgr  = _scale_to_height(ref_bgr, target_h)
    insp_bgr = _scale_to_height(insp_bgr, target_h)

    separator = np.full((target_h, separator_width, 3), 60, dtype=np.uint8)
    composite = np.concatenate([ref_bgr, separator, insp_bgr], axis=1)

    # Label bar above the panels
    label_h = 18
    label_bar = np.full((label_h, composite.shape[1], 3), 40, dtype=np.uint8)

    cv2.putText(
        label_bar, "Reference",
        (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
    )
    insp_x = ref_bgr.shape[1] + separator_width + 4
    cv2.putText(
        label_bar, "Inspection",
        (insp_x, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _WHITE, 1, cv2.LINE_AA,
    )
    if part_id:
        label_w = composite.shape[1]
        cv2.putText(
            label_bar, part_id,
            (label_w // 2 - 20, 13), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, _AMBER, 1, cv2.LINE_AA,
        )

    return np.concatenate([label_bar, composite], axis=0)
