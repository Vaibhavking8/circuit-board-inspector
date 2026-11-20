"""
comparison/template.py — template match signal for per-part presence scoring.

Uses cv2.matchTemplate (TM_CCOEFF_NORMED) to find the best matching location
within a neighbour-aware search window, then returns the peak correlation score.

Score is in [-1, 1] (TM_CCOEFF_NORMED range). Higher = more likely Present.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from boardinspect.comparison.interface import SignalResult
from boardinspect.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# TM_CCOEFF_NORMED is the default; names map to cv2 constants
_METHOD_MAP: dict[str, int] = {
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR_NORMED":  cv2.TM_CCORR_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
}


class TemplateMatchSignal:
    """Template correlation score between a reference crop and an inspection search window.

    Implements the Comparator protocol.

    The score is the peak normalised cross-correlation (TM_CCOEFF_NORMED) of the
    reference crop slid over the search window. High score → part present at the
    expected location; low or negative score → part absent or heavily changed.
    """

    @property
    def name(self) -> str:
        return "template"

    def compare(
        self,
        ref_crop: np.ndarray,
        search_region: np.ndarray,
        config: Config = DEFAULT_CONFIG,
    ) -> SignalResult:
        """Find the best match of *ref_crop* within *search_region*.

        Args:
            ref_crop:      Grayscale reference crop (uint8, H×W).
            search_region: Grayscale padded search window (uint8, H'×W').
                           H' ≥ H and W' ≥ W.
            config:        Config instance.

        Returns:
            SignalResult with name='template', value in [-1, 1], and location
            as (dx, dy) offset of the best match from the window centre.
        """
        comp = config.comparison
        method_name = comp.template_method
        method = _METHOD_MAP.get(method_name, cv2.TM_CCOEFF_NORMED)

        ref_h, ref_w = ref_crop.shape[:2]
        region_h, region_w = search_region.shape[:2]

        # Search region must be at least as large as the template
        if region_h < ref_h or region_w < ref_w:
            logger.warning(
                "TemplateMatch: search region (%dx%d) smaller than ref crop (%dx%d); "
                "returning score 0.",
                region_w, region_h, ref_w, ref_h,
            )
            return SignalResult(name=self.name, value=0.0, location=(0, 0))

        result_map = cv2.matchTemplate(search_region, ref_crop, method)

        if method == cv2.TM_SQDIFF_NORMED:
            # For SQDIFF the minimum is the best match; invert so higher = better
            min_val, _, min_loc, _ = cv2.minMaxLoc(result_map)
            best_x, best_y = min_loc
            score = float(1.0 - min_val)
        else:
            _, max_val, _, max_loc = cv2.minMaxLoc(result_map)
            best_x, best_y = max_loc
            score = float(max_val)

        # Optional sub-pixel refinement using a parabolic fit [EXPERIMENT]
        if comp.subpixel_refinement and result_map.size > 1:
            score, best_x, best_y = _subpixel_refine(
                result_map, best_x, best_y, score, method
            )

        # Express location as offset from the window centre
        # (where the part is expected to be when the board is well aligned)
        expected_x = (region_w - ref_w) // 2
        expected_y = (region_h - ref_h) // 2
        dx = best_x - expected_x
        dy = best_y - expected_y

        logger.debug(
            "TemplateMatch: score=%.3f, best_loc=(%d, %d), offset=(%d, %d).",
            score, best_x, best_y, dx, dy,
        )
        return SignalResult(name=self.name, value=score, location=(int(dx), int(dy)))


def _subpixel_refine(
    result_map: np.ndarray,
    px: int,
    py: int,
    score: float,
    method: int,
) -> tuple[float, int, int]:
    """Parabolic sub-pixel refinement around the peak in result_map.

    Fits independent 1-D parabolas along x and y through the three samples
    centred on the integer peak. The refined score is the interpolated peak
    value; the returned integer location is unchanged (sub-pixel displacement
    is a refinement of the score, not of the reported offset). [EXPERIMENT]

    Args:
        result_map: 2-D correlation map from cv2.matchTemplate.
        px, py:     Integer peak location in result_map.
        score:      Current best score at (px, py).
        method:     cv2 match method constant (used to handle SQDIFF sign).

    Returns:
        (refined_score, px, py) — location integers are unchanged.
    """
    h, w = result_map.shape
    if 0 < px < w - 1 and 0 < py < h - 1:
        v_xm = float(result_map[py, px - 1])
        v_x0 = float(result_map[py, px])
        v_xp = float(result_map[py, px + 1])
        v_ym = float(result_map[py - 1, px])
        v_yp = float(result_map[py + 1, px])

        denom_x = v_xm - 2.0 * v_x0 + v_xp
        denom_y = v_ym - 2.0 * v_x0 + v_yp

        sub_x = -0.5 * (v_xp - v_xm) / denom_x if abs(denom_x) > 1e-9 else 0.0
        sub_y = -0.5 * (v_yp - v_ym) / denom_y if abs(denom_y) > 1e-9 else 0.0

        refined_score = v_x0 + 0.5 * (v_xp - v_xm) * sub_x
        return float(refined_score), px, py

    return score, px, py


# Module-level singleton — the pipeline uses this instance by default
TEMPLATE_MATCH_SIGNAL = TemplateMatchSignal()
