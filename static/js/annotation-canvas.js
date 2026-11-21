/**
 * annotation-canvas.js — reference image annotation with named part boxes.
 *
 * Renders the reference image on #annotation-canvas and lets the user draw,
 * select, and delete named bounding boxes (parts). All boxes are stored in
 * state.js as normalised [0,1] coordinates so they scale correctly on resize.
 *
 * Visual layers (composited on the single canvas bitmap):
 *   1. Reference image scaled to fill the canvas.
 *   2. Committed part boxes, labels, and corner handles for the selection.
 *   3. In-progress rubber-band box while the user is dragging.
 *
 * @module annotation-canvas
 */

import { normalizedToPixel } from './coordinate-utils.js';
import {
  getParts, setParts,
  getReferenceSize, getReferenceId,
  getCurrentStep,
  subscribe,
} from './state.js';

// ── DOM references ─────────────────────────────────────────────────────────
const canvas      = document.getElementById('annotation-canvas');
const partsList   = document.getElementById('parts-list');
const partsCount  = document.getElementById('parts-count');
const partsHint   = document.getElementById('parts-hint');
const annotateErr = document.getElementById('annotate-error');   // existing in HTML

// ── Canvas API guard ───────────────────────────────────────────────────────
if (canvas && !canvas.getContext) {
  const msg = document.createElement('p');
  msg.className   = 'canvas-unsupported error-banner';
  msg.textContent =
    'Your browser does not support the canvas API. ' +
    'Please use a modern browser (Chrome, Firefox, Edge, or Safari).';
  canvas.replaceWith(msg);
}

/** @type {CanvasRenderingContext2D|null} */
const ctx = canvas ? canvas.getContext('2d') : null;

// ── Module-local state ─────────────────────────────────────────────────────
/** @type {HTMLImageElement|null} loaded reference image */
let _refImage  = null;
/** Width of the original reference image in pixels */
let _refWidth  = 0;
/** Height of the original reference image in pixels */
let _refHeight = 0;

/** @type {string|null} id of the currently selected part */
let _selectedId = null;

// In-progress draw gesture
let _drawing = false;
/** @type {{x:number,y:number}|null} */
let _drawStart = null;
/** @type {{x:number,y:number}|null} */
let _drawEnd   = null;

/** Minimum side length for an accepted box, measured in reference-image pixels. */
const MIN_BOX_PX = 8;

/** Timer handle for auto-dismissing the inline error. */
let _errorTimer = null;

// ── Inline error display ───────────────────────────────────────────────────

/**
 * Show a short validation message in the #annotate-error banner.
 * Auto-dismisses after 4 s.
 * @param {string} message
 */
function showError(message) {
  if (!annotateErr) {
    // Fallback: browsers that somehow don't have the element
    alert(message);
    return;
  }
  annotateErr.textContent = message;
  annotateErr.classList.remove('hidden');
  clearTimeout(_errorTimer);
  _errorTimer = setTimeout(() => {
    annotateErr.classList.add('hidden');
    annotateErr.textContent = '';
  }, 4000);
}

function clearError() {
  clearTimeout(_errorTimer);
  if (annotateErr) {
    annotateErr.classList.add('hidden');
    annotateErr.textContent = '';
  }
}

// ── Coordinate helpers ─────────────────────────────────────────────────────

/**
 * Returns the canvas element's bounding rect (safe even before layout paint).
 * @returns {DOMRect}
 */
function canvasRect() {
  return canvas ? canvas.getBoundingClientRect() : new DOMRect(0, 0, 1, 1);
}

/**
 * Convert a mouse or touch event to canvas-internal pixel coordinates,
 * accounting for any CSS scaling between the element and its bitmap.
 * @param {MouseEvent|TouchEvent} e
 * @returns {{x:number, y:number}}
 */
function eventToCanvas(e) {
  const rect    = canvasRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: (clientX - rect.left) * (canvas.width  / rect.width),
    y: (clientY - rect.top)  * (canvas.height / rect.height),
  };
}

/**
 * Same as eventToCanvas but reads from changedTouches (touchend events).
 * @param {Touch} touch
 * @returns {{x:number, y:number}}
 */
function changedTouchToCanvas(touch) {
  const rect = canvasRect();
  return {
    x: (touch.clientX - rect.left) * (canvas.width  / rect.width),
    y: (touch.clientY - rect.top)  * (canvas.height / rect.height),
  };
}

/**
 * Convert normalised [0,1] coordinates to canvas pixel coordinates.
 * @param {number} nx
 * @param {number} ny
 * @returns {[number, number]}
 */
function normToCanvas(nx, ny) {
  return [nx * canvas.width, ny * canvas.height];
}

// ── Rendering ──────────────────────────────────────────────────────────────

function render() {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Layer 1: reference image
  if (_refImage) {
    ctx.drawImage(_refImage, 0, 0, canvas.width, canvas.height);
  } else {
    // Placeholder while image loads
    ctx.fillStyle = '#1a1f2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  const parts = getParts();

  // Layer 2: committed part boxes
  for (const part of parts) {
    const [nx, ny, nw, nh] = part.box;
    const [cx, cy] = normToCanvas(nx, ny);
    const cw       = nw * canvas.width;
    const ch       = nh * canvas.height;
    const selected = part.id === _selectedId;

    // Box stroke
    ctx.strokeStyle = selected ? '#1565c0' : '#e65100';
    ctx.lineWidth   = selected ? 2.5 : 1.5;
    ctx.strokeRect(cx, cy, cw, ch);

    // Label pill above the box
    const fontSize = Math.max(10, Math.min(14, cw / 5));
    ctx.font = `bold ${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
    const textW = ctx.measureText(part.name).width;
    const padX  = 4;
    const padY  = 2;
    const pillH = fontSize + padY * 2;
    const pillW = textW + padX * 2;
    // Keep the pill inside the canvas vertically
    const pillY = cy >= pillH + 2 ? cy - pillH - 1 : cy + 1;

    ctx.fillStyle = selected ? '#1565c0' : '#e65100';
    ctx.fillRect(cx, pillY, pillW, pillH);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(part.name, cx + padX, pillY + fontSize + padY - 1);

    // Corner handles on the selected box
    if (selected) {
      const corners = [
        [cx,      cy],
        [cx + cw, cy],
        [cx,      cy + ch],
        [cx + cw, cy + ch],
      ];
      ctx.fillStyle = '#1565c0';
      for (const [hx, hy] of corners) {
        ctx.fillRect(hx - 4, hy - 4, 8, 8);
      }
    }
  }

  // Layer 3: in-progress rubber-band box
  if (_drawing && _drawStart && _drawEnd) {
    const x = Math.min(_drawStart.x, _drawEnd.x);
    const y = Math.min(_drawStart.y, _drawEnd.y);
    const w = Math.abs(_drawEnd.x - _drawStart.x);
    const h = Math.abs(_drawEnd.y - _drawStart.y);
    ctx.strokeStyle = '#2e7d32';
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}

// ── Sidebar parts list ─────────────────────────────────────────────────────

function renderPartsList() {
  if (!partsList) return;
  const parts = getParts();

  if (partsCount) partsCount.textContent = String(parts.length);
  if (partsHint)  partsHint.classList.toggle('hidden', parts.length > 0);

  partsList.innerHTML = '';
  for (const part of parts) {
    const li = document.createElement('li');
    li.className      = 'part-item' + (part.id === _selectedId ? ' selected' : '');
    li.dataset.partId = part.id;

    const nameSpan = document.createElement('span');
    nameSpan.className   = 'part-name';
    nameSpan.title       = part.name;
    nameSpan.textContent = part.name;

    const delBtn = document.createElement('button');
    delBtn.className   = 'part-delete btn-ghost btn-sm';
    delBtn.setAttribute('aria-label', `Delete ${part.name}`);
    delBtn.dataset.id  = part.id;
    delBtn.textContent = '✕';

    li.appendChild(nameSpan);
    li.appendChild(delBtn);

    // Click the row to select the part
    li.addEventListener('click', (e) => {
      if (e.target === delBtn) return;
      _selectedId = part.id;
      render();
      renderPartsList();
    });

    // Click the delete button
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      deletePart(part.id);
    });

    partsList.appendChild(li);
  }
}

// ── CRUD ───────────────────────────────────────────────────────────────────

/**
 * Check that a name is not already used by an existing part (case-insensitive).
 * @param {string} name
 * @returns {boolean}
 */
function isUniqueName(name) {
  return !getParts().some(p => p.name.toLowerCase() === name.toLowerCase());
}

/**
 * Finalise a draw gesture: validate size and name, then commit to state.
 *
 * @param {number} x1 canvas px — first corner x
 * @param {number} y1 canvas px — first corner y
 * @param {number} x2 canvas px — opposite corner x
 * @param {number} y2 canvas px — opposite corner y
 */
function commitBox(x1, y1, x2, y2) {
  if (!canvas) return;

  // Normalise to top-left origin in [0,1]
  const nx = Math.min(x1, x2) / canvas.width;
  const ny = Math.min(y1, y2) / canvas.height;
  const nw = Math.abs(x2 - x1) / canvas.width;
  const nh = Math.abs(y2 - y1) / canvas.height;

  // Minimum size check in reference-image pixels (not canvas pixels)
  const refPw = Math.round(nw * _refWidth);
  const refPh = Math.round(nh * _refHeight);
  if (refPw < MIN_BOX_PX || refPh < MIN_BOX_PX) {
    showError(
      `Box is too small (${refPw}×${refPh} px). ` +
      `Minimum is ${MIN_BOX_PX}×${MIN_BOX_PX} px — try drawing a larger box.`
    );
    return;
  }

  // Name prompt (synchronous; can be replaced with a modal later)
  let name = window.prompt('Enter a name for this part:');
  if (name === null) return;        // user cancelled
  name = name.trim();

  if (!name) {
    showError('Part name cannot be empty.');
    return;
  }
  if (!isUniqueName(name)) {
    showError(`A part named "${name}" already exists. Part names must be unique.`);
    return;
  }

  clearError();

  const id    = 'p' + Date.now().toString(36);
  const parts = [...getParts(), { id, name, box: [nx, ny, nw, nh] }];
  setParts(parts);       // triggers subscribe('parts') → render + renderPartsList
  _selectedId = id;
  render();
  renderPartsList();
}

/**
 * Remove a part by id.
 * @param {string} id
 */
function deletePart(id) {
  const parts = getParts().filter(p => p.id !== id);
  if (_selectedId === id) _selectedId = null;
  setParts(parts);       // triggers subscribe('parts') → render + renderPartsList
}

// ── Hit testing ────────────────────────────────────────────────────────────

/**
 * Return the id of the topmost part whose box contains (cx, cy), or null.
 * Parts are tested in reverse insertion order so the most recently added
 * part is "on top" visually.
 * @param {number} cx  canvas px
 * @param {number} cy  canvas px
 * @returns {string|null}
 */
function hitTest(cx, cy) {
  const parts = getParts();
  for (let i = parts.length - 1; i >= 0; i--) {
    const [nx, ny, nw, nh] = parts[i].box;
    const [bx, by] = normToCanvas(nx, ny);
    const bw = nw * canvas.width;
    const bh = nh * canvas.height;
    if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh) {
      return parts[i].id;
    }
  }
  return null;
}

// ── Mouse event handlers ───────────────────────────────────────────────────

canvas?.addEventListener('mousedown', (e) => {
  // Ignore if modifier keys are held (let browser do its thing)
  if (e.button !== 0) return;
  const { x, y } = eventToCanvas(e);
  const hit = hitTest(x, y);
  if (hit) {
    // Click inside an existing box — select it, do not start drawing
    _selectedId = hit;
    render();
    renderPartsList();
  } else {
    // Click on empty area — start a new draw gesture
    _drawing    = true;
    _drawStart  = { x, y };
    _drawEnd    = { x, y };
    _selectedId = null;
  }
  clearError();
});

canvas?.addEventListener('mousemove', (e) => {
  if (!_drawing) return;
  _drawEnd = eventToCanvas(e);
  render();
});

canvas?.addEventListener('mouseup', (e) => {
  if (!_drawing) return;
  _drawing = false;
  const { x, y } = eventToCanvas(e);
  const start = _drawStart;
  _drawStart = null;
  _drawEnd   = null;
  commitBox(start.x, start.y, x, y);
  render();
});

canvas?.addEventListener('mouseleave', () => {
  if (_drawing) {
    _drawing   = false;
    _drawStart = null;
    _drawEnd   = null;
    render();
  }
});

// ── Touch event handlers ───────────────────────────────────────────────────

canvas?.addEventListener('touchstart', (e) => {
  e.preventDefault();
  const { x, y } = eventToCanvas(e);
  _drawing   = true;
  _drawStart = { x, y };
  _drawEnd   = { x, y };
  _selectedId = null;
}, { passive: false });

canvas?.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (!_drawing) return;
  _drawEnd = eventToCanvas(e);
  render();
}, { passive: false });

canvas?.addEventListener('touchend', (e) => {
  e.preventDefault();
  if (!_drawing) return;
  _drawing = false;
  const touch = e.changedTouches[0];
  const { x, y } = changedTouchToCanvas(touch);
  const start = _drawStart;
  _drawStart = null;
  _drawEnd   = null;
  commitBox(start.x, start.y, x, y);
  render();
}, { passive: false });

// ── Keyboard handler ───────────────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Delete' && e.key !== 'Backspace') return;
  if (!_selectedId) return;
  // Don't intercept keyboard events inside form controls
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  deletePart(_selectedId);
});

// ── Window resize ──────────────────────────────────────────────────────────

window.addEventListener('resize', () => {
  if (!_refImage || !canvas) return;
  _fitCanvas();
  render();
});

// ── State subscriptions ────────────────────────────────────────────────────

// When parts change from any source (e.g. storage load), re-render.
subscribe('parts', () => {
  render();
  renderPartsList();
});

// Auto-initialise when the user reaches Step 2 (annotation step).
subscribe('currentStep', (step) => {
  if (step !== 2) return;
  const size = getReferenceSize();
  if (!size) return;
  // The reference image object URL is already in the DOM preview element.
  const refImg = /** @type {HTMLImageElement|null} */ (
    document.getElementById('reference-img')
  );
  if (!refImg?.src) return;
  initCanvas(getReferenceId(), size.width, size.height, refImg.src);
});

// ── Internal helpers ───────────────────────────────────────────────────────

/**
 * Size the canvas bitmap to match the reference image aspect ratio,
 * capped at the container's current client width.
 */
function _fitCanvas() {
  if (!canvas || !_refWidth || !_refHeight) return;
  const container = canvas.parentElement;
  const maxW      = container ? container.clientWidth : 800;
  const scale     = Math.min(1, maxW / _refWidth);
  canvas.width    = Math.round(_refWidth  * scale);
  canvas.height   = Math.round(_refHeight * scale);
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Initialise (or re-initialise) the canvas with a reference image.
 *
 * Sizes the canvas bitmap to fit the container, loads the image, and
 * re-renders all parts. Called automatically when the user reaches Step 2,
 * and can also be called by external code (e.g. after loading a saved setup).
 *
 * @param {string|null} referenceId  Reference setup id (kept for API symmetry;
 *                                   not used for drawing).
 * @param {number}      width        Original reference image width in pixels.
 * @param {number}      height       Original reference image height in pixels.
 * @param {string}      imageSrc     Object URL or data URL of the reference image.
 */
export function initCanvas(referenceId, width, height, imageSrc) {
  _refWidth  = width;
  _refHeight = height;
  _selectedId = null;

  _fitCanvas();

  const img   = new Image();
  img.onload  = () => {
    _refImage = img;
    render();
    renderPartsList();
  };
  img.onerror = () => {
    console.error('[annotation-canvas] Failed to load reference image:', imageSrc);
    showError('Could not load the reference image. Please try uploading it again.');
  };
  img.src = imageSrc;
}

/**
 * Programmatically deselect the current part.
 * Call this when the user advances to Step 3 so no box appears highlighted.
 */
export function clearSelection() {
  _selectedId = null;
  render();
  renderPartsList();
}

/**
 * Return a snapshot of the current parts array (normalised coords).
 * Callers that need live updates should subscribe via state.js instead.
 * @returns {Array<{id:string, name:string, box:[number,number,number,number]}>}
 */
export function getCanvasParts() {
  return getParts();
}
