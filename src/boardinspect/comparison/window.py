"""
comparison/window.py — search window computation for template matching.

The search window for a part is a padded rectangle around its expected
location in the inspection image. The padding is capped so the window
cannot extend into a neighbouring part's region.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from boardinspect.config import Config, DEFAULT_CONFIG


@dataclass
class SearchWindow:
    """Pixel-coordinate search window on the inspection image.

    All coordinates are in pixels on the comparison-resolution image.
    """
    x: int      # left edge of the window
    y: int      # top edge of the window
    w: int      # width
    h: int      # height


def _nearest_neighbour_distance(
    box: tuple[int, int, int, int],
    other_boxes: list[tuple[int, int, int, int]],
) -> float:
    """Minimum distance from the edge of *box* to the edge of any box in *other_boxes*.

    Distance is measured as the minimum gap between the nearest edges of the
    two boxes (not centre-to-centre). Returns infinity if other_boxes is empty.

    Args:
        box:         (x, y, w, h) in pixels.
        other_boxes: List of (x, y, w, h) in pixels for all other parts.

    Returns:
        Minimum edge-to-edge distance in pixels.
    """
    bx, by, bw, bh = box
    bx2, by2 = bx + bw, by + bh

    min_dist = math.inf
    for ox, oy, ow, oh in other_boxes:
        ox2, oy2 = ox + ow, oy + oh
        # Horizontal gap: gap is 0 when boxes overlap or touch in x
        if bx2 <= ox:
            dx = ox - bx2
        elif ox2 <= bx:
            dx = bx - ox2
        else:
            dx = 0
        # Vertical gap: gap is 0 when boxes overlap or touch in y
        if by2 <= oy:
            dy = oy - by2
        elif oy2 <= by:
            dy = by - oy2
        else:
            dy = 0
        dist = math.sqrt(dx * dx + dy * dy)
        min_dist = min(min_dist, dist)

    return min_dist


def compute_search_window(
    part_box_px: tuple[int, int, int, int],
    other_boxes_px: list[tuple[int, int, int, int]],
    image_w: int,
    image_h: int,
    config: Config = DEFAULT_CONFIG,
) -> SearchWindow:
    """Compute the search window for one part on the inspection image.

    Padding rules:
    1. Base padding = config.comparison.search_window_padding_fraction × (box w, box h).
    2. Cap half-padding to config.comparison.search_window_neighbour_limit_fraction
       of the nearest-neighbour edge-to-edge distance.
    3. Clamp to image boundaries.
    4. Minimum window size equals the part box itself (never shrink below it).

    Args:
        part_box_px:    (x, y, w, h) of this part in inspection-frame pixels.
        other_boxes_px: (x, y, w, h) of all other parts in inspection-frame pixels.
        image_w:        Inspection image width in pixels.
        image_h:        Inspection image height in pixels.
        config:         Config instance.

    Returns:
        SearchWindow giving the padded region to search.
    """
    comp = config.comparison
    bx, by, bw, bh = part_box_px

    # Rule 1: base padding
    pad_x = comp.search_window_padding_fraction * bw
    pad_y = comp.search_window_padding_fraction * bh

    # Rule 2: cap by neighbour distance
    if other_boxes_px:
        nn_dist = _nearest_neighbour_distance(part_box_px, other_boxes_px)
        max_half = comp.search_window_neighbour_limit_fraction * nn_dist
        pad_x = min(pad_x, max_half)
        pad_y = min(pad_y, max_half)

    # Rule 3: build window and clamp to image bounds
    wx = int(max(0, bx - pad_x))
    wy = int(max(0, by - pad_y))
    wx2 = int(min(image_w, bx + bw + pad_x))
    wy2 = int(min(image_h, by + bh + pad_y))

    # Rule 4: window must be at least as large as the part box
    return SearchWindow(x=wx, y=wy, w=max(bw, wx2 - wx), h=max(bh, wy2 - wy))
