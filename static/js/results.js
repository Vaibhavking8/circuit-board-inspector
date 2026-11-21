/**
 * results.js — inspection upload, result cards, sensitivity re-evaluation, debug.
 *
 * Responsibilities:
 *   - Drag-and-drop / file-input handling for inspection images (max 3).
 *   - Per-image loading skeletons that are replaced as each job completes.
 *   - Drawing annotated canvases: photo with coloured boxes and text labels
 *     for every non-present part (R8.5: text + icon, never colour alone).
 *   - Sensitivity slider re-evaluation (POST /re-evaluate, then re-render).
 *   - Debug panel showing per-part signals, thresholds and margins.
 *
 * main.js owns the debug-toggle button click and the slider input display.
 * This module only adds the slider "change" listener for re-evaluation.
 * @module results
 */

import { request, requestFile, pollJob } from './api-client.js';
import {
  getReferenceId, getSensitivity, setSensitivity,
  getResults, setResults, addResult, subscribe,
} from './state.js';

const MAX_IMAGES = 3;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone     = document.getElementById('inspect-drop-zone');
const fileInput    = document.getElementById('inspect-file-input');
const resultsGrid  = document.getElementById('inspect-results');
const errorDiv     = document.getElementById('inspect-error');
const sensSlider   = document.getElementById('sensitivity-slider');
const debugContent = document.getElementById('debug-content');

// ── Status display config (text + icon, R8.5) ────────────────────────────────
// Maps API status strings to card CSS modifier, banner CSS class, label, icon.
const STATUS_CONFIG = {
  ok: {
    cardMod: 'ok',
    bannerCls: 'status-banner-ok',
    icon: '✅',
    label: 'OK — No missing parts',
  },
  missing_parts: {
    cardMod: 'missing',           // → result-card.status-missing (CSS rule)
    bannerCls: 'status-banner-missing',
    icon: '🔴',
    label: 'Missing Parts',
  },
  could_not_verify: {
    cardMod: 'uncertain',
    bannerCls: 'status-banner-uncertain',
    icon: '⚠️',
    label: 'Could Not Verify',
  },
};

// ── Error helper ──────────────────────────────────────────────────────────────
function showError(msg) {
  if (!errorDiv) return;
  errorDiv.textContent = msg;
  errorDiv.classList.remove('hidden');
}
function clearError() {
  errorDiv?.classList.add('hidden');
}

// ── Canvas overlays ───────────────────────────────────────────────────────────

/**
 * Draw coloured boxes and text labels for every non-present part.
 * Colour is always accompanied by text (R8.5).
 * inspection_box is already in pixels relative to the original image dimensions.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} result  ImageResultResponse
 */
function _drawOverlays(ctx, result) {
  for (const pr of result.part_results) {
    if (pr.status === 'present') continue;

    const [bx, by, bw, bh] = pr.inspection_box;
    const colour = pr.status === 'missing' ? '#c62828' : '#e65100';

    // Box outline
    ctx.strokeStyle = colour;
    ctx.lineWidth   = 3;
    ctx.strokeRect(bx, by, bw, bh);

    // Label background + text — text ensures the state is communicated
    // even if the reader cannot distinguish colour (R8.5).
    const label = pr.status === 'missing'
      ? `\u2717 ${pr.part_id}`    // ✗ PartName
      : `? ${pr.part_id}`;
    ctx.font      = 'bold 14px sans-serif';
    const tw      = ctx.measureText(label).width;
    const labelH  = 20;

    // Place the label above the box; if the box is at the top, place below.
    const labelY = by >= labelH ? by : by + bh;

    ctx.fillStyle = colour;
    ctx.fillRect(bx, labelY - labelH, tw + 8, labelH);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, bx + 4, labelY - 4);
  }
}

// ── Card rendering ────────────────────────────────────────────────────────────

/**
 * Render a completed result into the given wrapper element.
 * The wrapper's contents are fully replaced.
 *
 * @param {HTMLElement} wrapper
 * @param {object}      result      ImageResultResponse
 * @param {string}      imageUrl    Object URL for the inspection image
 * @param {number}      index       0-based card index (for aria-label)
 */
function _renderCard(wrapper, result, imageUrl, index) {
  const cfg         = STATUS_CONFIG[result.status] ?? STATUS_CONFIG.could_not_verify;
  const missingParts = result.part_results.filter(p => p.status === 'missing');
  const runtimeStr  = typeof result.runtime_ms === 'number'
    ? `${result.runtime_ms.toFixed(0)} ms`
    : '';

  wrapper.innerHTML = `
    <div class="result-card status-${cfg.cardMod}" data-index="${index}">

      <div class="result-status-banner ${cfg.bannerCls}" role="status">
        <span class="status-icon" aria-hidden="true">${cfg.icon}</span>
        <span class="status-label">${cfg.label}</span>
        ${runtimeStr ? `<span class="runtime-label" style="margin-left:auto;font-size:.8em;opacity:.8">${runtimeStr}</span>` : ''}
      </div>

      <div class="result-canvas-wrap">
        <canvas class="result-canvas"
                aria-label="Inspection image ${index + 1} with overlays"></canvas>
      </div>

      <div class="result-body">
        ${missingParts.length > 0 ? `
          <p style="font-weight:600;color:var(--color-missing);margin-bottom:.5rem">
            Missing: ${missingParts.map(p => p.part_id).join(', ')}
          </p>
          <ul class="missing-parts-list">
            ${missingParts.map(p => `
              <li class="missing-part-entry">
                <span class="missing-part-icon" aria-hidden="true">📌</span>
                <span>${p.part_id}</span>
              </li>`).join('')}
          </ul>` : ''}
        ${result.messages?.length ? `
          <p style="font-size:.8em;color:var(--color-text-muted);margin-top:.5rem">
            ${result.messages.join(' ')}
          </p>` : ''}
      </div>
    </div>`;

  // Draw overlays once the image is decoded.
  const canvas = wrapper.querySelector('.result-canvas');
  if (!canvas || !imageUrl) return;

  const img = new Image();
  img.onload = () => {
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    _drawOverlays(ctx, result);
    // Release the object URL — the pixels are now in the canvas.
    URL.revokeObjectURL(imageUrl);
  };
  img.src = imageUrl;
}

// ── Loading skeleton ──────────────────────────────────────────────────────────

function _addLoadingCard(index) {
  const wrap = document.createElement('div');
  wrap.id = `result-wrap-${index}`;
  wrap.innerHTML = `
    <div class="result-card loading">
      <div class="result-status-banner">
        <span class="status-label">Inspecting\u2026</span>
      </div>
      <div class="result-body">
        <div class="skeleton" style="height:120px"></div>
        <div class="skeleton" style="width:60%;margin-top:.5rem"></div>
      </div>
    </div>`;
  resultsGrid?.appendChild(wrap);
  return wrap;
}

// ── Error card ────────────────────────────────────────────────────────────────

function _renderErrorCard(wrapper, message) {
  wrapper.innerHTML = `
    <div class="result-card status-uncertain">
      <div class="result-status-banner status-banner-uncertain" role="alert">
        <span class="status-icon" aria-hidden="true">\u26a0\ufe0f</span>
        <span class="status-label">Error</span>
      </div>
      <div class="result-body"><p>${message}</p></div>
    </div>`;
}

// ── Per-job state (wrapper + imageUrl, indexed by position) ──────────────────
// Needed so re-evaluate can re-draw canvases.
const _jobState = [];   // [{ wrapper, imageUrl, result }]

// ── Upload & inspect ──────────────────────────────────────────────────────────

/**
 * Upload inspection files, start a background job per file, and render each
 * result card as soon as its job completes (R12.2 — results appear per image).
 *
 * @param {FileList|File[]} files
 */
async function inspectFiles(files) {
  clearError();

  const refId = getReferenceId();
  if (!refId) {
    showError('No reference image loaded. Go to Step 1.');
    return;
  }

  const fileArr = Array.from(files);
  if (fileArr.length > MAX_IMAGES) {
    showError(`Only the first ${MAX_IMAGES} images will be inspected; the rest were ignored.`);
  }
  const limited = fileArr.slice(0, MAX_IMAGES);
  if (!limited.length) return;

  if (!resultsGrid) return;
  resultsGrid.innerHTML = '';
  setResults([]);
  _jobState.length = 0;

  const sensitivity = Number(sensSlider?.value ?? 50) / 100;

  for (let i = 0; i < limited.length; i++) {
    const file     = limited[i];
    const wrapper  = _addLoadingCard(i);
    const imageUrl = URL.createObjectURL(file);
    // Reserve the slot so re-evaluate indices stay aligned.
    _jobState.push({ wrapper, imageUrl: null, result: null });

    // Fire-and-forget per image so all three start uploading immediately.
    (async (idx) => {
      try {
        const job = await requestFile(
          `/api/reference/${refId}/inspect`,
          file,
          'file',
          { sensitivity: sensitivity.toFixed(2) },
        );
        const completed = await pollJob(job.job_id, () => {});
        const result    = completed.result;

        // Store result and image URL for later re-render on sensitivity change.
        _jobState[idx].result   = result;
        _jobState[idx].imageUrl = imageUrl;

        addResult(result);
        _renderCard(wrapper, result, imageUrl, idx);
        _renderDebugPanel(getResults());
      } catch (err) {
        URL.revokeObjectURL(imageUrl);
        _renderErrorCard(wrapper, err.message);
      }
    })(i);
  }
}

// ── Sensitivity re-evaluate ───────────────────────────────────────────────────

/**
 * Re-apply the decision engine at the new sensitivity without re-uploading.
 * Updates each visible card in place.
 */
async function reEvaluate() {
  const results = getResults();
  if (!results.length) return;

  const refId = getReferenceId();
  if (!refId) return;

  const sensitivity = Number(sensSlider?.value ?? 50) / 100;
  const updated     = [];

  for (let i = 0; i < _jobState.length; i++) {
    const slot   = _jobState[i];
    const result = slot.result;
    if (!result) { updated.push(result); continue; }

    try {
      const alignStatus = result.alignment?.transform_type === 'identity'
        ? 'identity'
        : (result.alignment?.transform_type ?? 'aligned');

      // Retrieve image dimensions from the canvas if available; fall back to
      // a wide-screen default that only affects the inspection_box mapping —
      // the signals stored in part_results are independent of image size.
      const canvas = slot.wrapper?.querySelector('.result-canvas');
      const iw = canvas?.width  || 1920;
      const ih = canvas?.height || 1080;

      const resp = await request('POST', `/api/reference/${refId}/re-evaluate`, {
        sensitivity,
        part_results:    result.part_results,
        alignment_status: alignStatus,
        inspection_width:  iw,
        inspection_height: ih,
      });

      const newResult = { ...result, status: resp.status, part_results: resp.part_results };
      updated.push(newResult);

      // Update the slot so a subsequent re-evaluate uses the latest part_results.
      slot.result = newResult;

      // Re-render card in place: create a fresh object URL from the stored
      // canvas data since the original File is gone.
      if (canvas) {
        canvas.toBlob(blob => {
          if (!blob) return;
          const url = URL.createObjectURL(blob);
          _renderCard(slot.wrapper, newResult, url, i);
        }, 'image/png');
      }
    } catch {
      updated.push(result);    // keep old result on network error
    }
  }

  setResults(updated);
  _renderDebugPanel(updated);
}

// ── Debug panel ───────────────────────────────────────────────────────────────

function _renderDebugPanel(results) {
  if (!debugContent) return;

  const lines = results.flatMap((r, i) => {
    if (!r?.part_results) return [];
    return r.part_results.map(pr => {
      const sigParts = Object.entries(pr.signals ?? {})
        .map(([k, v]) => `${k}=${Number(v).toFixed(3)}`)
        .join(' ');
      const thr = Number(pr.threshold ?? 0).toFixed(3);
      const mrg = Number(pr.margin    ?? 0).toFixed(3);
      return `[img ${i + 1}] ${pr.part_id}: ${pr.status} | ${sigParts} | threshold=${thr} margin=${mrg}`;
    });
  });

  debugContent.textContent = lines.join('\n') || '(no results yet)';
}

subscribe('results', results => _renderDebugPanel(results));

// ── Event wiring ──────────────────────────────────────────────────────────────

// Drop zone — click delegates to hidden file input.
dropZone?.addEventListener('click', e => {
  if (e.target !== fileInput) fileInput?.click();
});
dropZone?.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput?.click();
  }
});

// File input change.
fileInput?.addEventListener('change', () => {
  if (fileInput.files?.length) inspectFiles(fileInput.files);
  fileInput.value = '';   // allow re-selecting the same file
});

// Drag-and-drop.
dropZone?.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone?.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});
dropZone?.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  if (e.dataTransfer?.files?.length) inspectFiles(e.dataTransfer.files);
});

// Sensitivity "change" triggers re-evaluation.
// The "input" listener (display update) is already in main.js.
sensSlider?.addEventListener('change', () => reEvaluate());

// Keep state in sync when slider moves.
sensSlider?.addEventListener('input', () => {
  setSensitivity(Number(sensSlider.value));
});

// ── Public API ────────────────────────────────────────────────────────────────

/** Called by main.js at start-up. All listeners are registered above. */
export function initResults() {}
