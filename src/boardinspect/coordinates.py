"""
coordinates.py — canonical coordinate converter for BoardInspect.

Persisted coordinates are normalised (x, y, w, h) in [0, 1] relative to the
reference image's original pixel dimensions, origin top-left.

Conversion rule (must match coordinate-utils.js exactly):
    pixel = round(normalised * size)   # for position and size
    clamped to [0, size]               # position never negative or beyond edge
    minimum size enforced              # w and h are at least min_size

This module has zero imports from the web framework, storage, or imaging layers.
"""
from __future__ import annotations


NormBox = tuple[float, float, float, float]   # (x, y, w, h) in [0, 1]
PixelBox = tuple[int, int, int, int]           # (x, y, w, h) in pixels


def normalized_to_pixel(
    box_norm: NormBox,
    width: int,
    height: int,
    min_size: int = 1,
) -> PixelBox:
    """Convert a normalised box to integer pixel coordinates.

    Conversion rule: pixel = round(normalised * size), then clamp to image
    bounds and enforce a minimum side length. This rule must remain identical
    to the browser mirror in ``coordinate-utils.js``.

    Args:
        box_norm:  (x, y, w, h) each in [0, 1], origin top-left.
        width:     Reference image width in pixels.
        height:    Reference image height in pixels.
        min_size:  Minimum pixel side length for w and h (default 1).

    Returns:
        (x, y, w, h) as integers, clamped to the image bounds.
    """
    xn, yn, wn, hn = box_norm
    x = int(round(xn * width))
    y = int(round(yn * height))
    w = int(round(wn * width))
    h = int(round(hn * height))

    # Enforce minimum size before clamping positions so that a tiny normalised
    # box still produces a visible region.
    w = max(w, min_size)
    h = max(h, min_size)

    # Clamp origin so the box starts inside the image.
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    # Clamp size so the box does not exceed the image boundary.
    w = min(w, width - x)
    h = min(h, height - y)

    # Re-enforce minimum after clamping: edge case where the box origin is
    # flush with the right/bottom boundary and min_size > available pixels.
    # We allow the clamped available space to win when it is smaller than
    # min_size (image is too small), so we don't go negative.
    w = max(w, min(min_size, width - x))
    h = max(h, min(min_size, height - y))

    return (x, y, w, h)


def pixel_to_normalized(
    box_px: PixelBox,
    width: int,
    height: int,
) -> NormBox:
    """Convert integer pixel coordinates to normalised [0, 1] form.

    Args:
        box_px:  (x, y, w, h) in pixels.
        width:   Image width in pixels.
        height:  Image height in pixels.

    Returns:
        (x, y, w, h) each in [0, 1], clamped.
    """
    x, y, w, h = box_px
    xn = max(0.0, min(x / width, 1.0))
    yn = max(0.0, min(y / height, 1.0))
    wn = max(0.0, min(w / width, 1.0))
    hn = max(0.0, min(h / height, 1.0))
    return (xn, yn, wn, hn)


def clamp_box(
    box_px: PixelBox,
    width: int,
    height: int,
    min_size: int = 1,
) -> PixelBox:
    """Clamp a pixel box to the image boundary and enforce a minimum size.

    Use this to sanitise boxes that arrive from the frontend or from an
    alignment transform that may push boxes slightly out of bounds.

    Args:
        box_px:   (x, y, w, h) in pixels.
        width:    Image width in pixels.
        height:   Image height in pixels.
        min_size: Minimum side length in pixels.

    Returns:
        Clamped (x, y, w, h).
    """
    x, y, w, h = box_px

    w = max(w, min_size)
    h = max(h, min_size)

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    w = min(w, width - x)
    h = min(h, height - y)

    # Allow the available space to win when it is smaller than min_size.
    w = max(w, min(min_size, width - x))
    h = max(h, min(min_size, height - y))

    return (x, y, w, h)
