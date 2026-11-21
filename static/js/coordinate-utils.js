/**
 * coordinate-utils.js — canonical coordinate converter (JS mirror of coordinates.py).
 *
 * Conversion rule (must match coordinates.py exactly):
 *   pixel = Math.round(normalised * size)   // for position and size
 *   clamped to [0, size]                    // position never outside image
 *   minimum size enforced                   // w and h are at least minSize
 *
 * All functions are pure (no I/O, no globals mutated).
 * @module coordinate-utils
 */

/**
 * Convert a normalised box [0,1] to integer pixel coordinates.
 *
 * Conversion rule: pixel = Math.round(normalised * size), then clamp to image
 * bounds and enforce a minimum side length. This rule must remain identical
 * to the Python mirror in ``coordinates.py``.
 *
 * @param {[number, number, number, number]} boxNorm  [xn, yn, wn, hn] in [0,1]
 * @param {number} width   Image width in pixels
 * @param {number} height  Image height in pixels
 * @param {number} [minSize=1]  Minimum pixel side length
 * @returns {[number, number, number, number]}  [x, y, w, h] as integers
 */
export function normalizedToPixel(boxNorm, width, height, minSize = 1) {
  const [xn, yn, wn, hn] = boxNorm;

  let x = Math.round(xn * width);
  let y = Math.round(yn * height);
  let w = Math.round(wn * width);
  let h = Math.round(hn * height);

  // Enforce minimum size before clamping positions so that a tiny normalised
  // box still produces a visible region.
  w = Math.max(w, minSize);
  h = Math.max(h, minSize);

  // Clamp origin so the box starts inside the image.
  x = Math.max(0, Math.min(x, width  - 1));
  y = Math.max(0, Math.min(y, height - 1));

  // Clamp size so the box does not exceed the image boundary.
  w = Math.min(w, width  - x);
  h = Math.min(h, height - y);

  // Re-enforce minimum after clamping: edge case where the box origin is
  // flush with the right/bottom boundary and minSize > available pixels.
  // Allow the clamped available space to win when it is smaller than
  // minSize (image is too small), so we don't go negative.
  w = Math.max(w, Math.min(minSize, width  - x));
  h = Math.max(h, Math.min(minSize, height - y));

  return [x, y, w, h];
}

/**
 * Convert integer pixel coordinates to normalised [0,1] form.
 *
 * @param {[number, number, number, number]} boxPx  [x, y, w, h] in pixels
 * @param {number} width   Image width in pixels
 * @param {number} height  Image height in pixels
 * @returns {[number, number, number, number]}  [xn, yn, wn, hn] in [0,1]
 */
export function pixelToNormalized(boxPx, width, height) {
  const [x, y, w, h] = boxPx;
  const xn = Math.max(0, Math.min(x / width,  1));
  const yn = Math.max(0, Math.min(y / height, 1));
  const wn = Math.max(0, Math.min(w / width,  1));
  const hn = Math.max(0, Math.min(h / height, 1));
  return [xn, yn, wn, hn];
}

/**
 * Clamp a pixel box to the image boundary and enforce a minimum size.
 *
 * Use this to sanitise boxes that arrive from the frontend or from an
 * alignment transform that may push boxes slightly out of bounds.
 *
 * @param {[number, number, number, number]} boxPx   [x, y, w, h] in pixels
 * @param {number} width    Image width in pixels
 * @param {number} height   Image height in pixels
 * @param {number} [minSize=1]  Minimum side length in pixels
 * @returns {[number, number, number, number]}  Clamped [x, y, w, h]
 */
export function clampBox(boxPx, width, height, minSize = 1) {
  let [x, y, w, h] = boxPx;

  w = Math.max(w, minSize);
  h = Math.max(h, minSize);

  x = Math.max(0, Math.min(x, width  - 1));
  y = Math.max(0, Math.min(y, height - 1));

  w = Math.min(w, width  - x);
  h = Math.min(h, height - y);

  // Allow the available space to win when it is smaller than minSize.
  w = Math.max(w, Math.min(minSize, width  - x));
  h = Math.max(h, Math.min(minSize, height - y));

  return [x, y, w, h];
}
