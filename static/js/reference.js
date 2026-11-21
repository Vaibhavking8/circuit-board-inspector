/**
 * reference.js — reference image upload, preview, and load saved setup.
 *
 * Handles drag-and-drop and click-to-browse for the reference image.
 * Validates file client-side (type, rough size) before uploading.
 * On success, updates state and shows the preview.
 * On error, shows a message and keeps the previous reference if any.
 * @module reference
 */

import { requestFile } from './api-client.js';
import {
  setReferenceId, setReferenceSize, getReferenceId, getParts, setParts,
} from './state.js';

const MAX_CLIENT_SIZE_MB = 30;  // soft client-side cap (server enforces 25 MB)
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp', 'image/tiff'];

// ── DOM references ──────────────────────────────────────────────────────────
const dropZone    = document.getElementById('reference-drop-zone');
const fileInput   = document.getElementById('reference-file-input');
const previewArea = document.getElementById('reference-preview');
const previewImg  = document.getElementById('reference-img');
const infoSpan    = document.getElementById('reference-info');
const errorDiv    = document.getElementById('reference-error');
const btnChange   = document.getElementById('btn-change-reference');
const btnNext     = document.getElementById('btn-next-to-annotate');

// ── Helpers ─────────────────────────────────────────────────────────────────

function showError(msg) {
  if (!errorDiv) return;
  errorDiv.textContent = msg;
  errorDiv.classList.remove('hidden');
}

function clearError() {
  if (errorDiv) errorDiv.classList.add('hidden');
}

function showPreview(src, label) {
  if (!previewImg || !previewArea) return;
  previewImg.src = src;
  if (infoSpan) infoSpan.textContent = label;
  previewArea.classList.remove('hidden');
  if (dropZone) dropZone.classList.add('hidden');
}

function validateFile(file) {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    throw new Error(`Unsupported file type "${file.type}". Please upload a JPEG, PNG, BMP, WebP, or TIFF image.`);
  }
  const mb = file.size / 1_048_576;
  if (mb > MAX_CLIENT_SIZE_MB) {
    throw new Error(`File is too large (${mb.toFixed(1)} MB). Maximum size is 25 MB.`);
  }
}

async function uploadFile(file) {
  clearError();
  try {
    validateFile(file);
  } catch (err) {
    showError(err.message);
    return;
  }

  // If parts exist, confirm before replacing
  if (getReferenceId() && getParts().length > 0) {
    if (!confirm('Replacing the reference image will clear all annotated parts. Continue?')) return;
    setParts([]);
  }

  // Show loading state
  if (dropZone) dropZone.setAttribute('aria-busy', 'true');

  try {
    const data = await requestFile('/api/reference', file);
    const { reference_id, width, height } = data;
    setReferenceId(reference_id);
    setReferenceSize({ width, height });

    // Show preview using object URL
    const url = URL.createObjectURL(file);
    showPreview(url, `${file.name} — ${width}×${height} px`);
    clearError();
  } catch (err) {
    showError(err.message);
    // Keep previous reference in state (do not reset)
  } finally {
    if (dropZone) dropZone.removeAttribute('aria-busy');
  }
}

// ── Event wiring ─────────────────────────────────────────────────────────────

// Click anywhere in drop zone → trigger file input
dropZone?.addEventListener('click', e => {
  if (e.target !== fileInput) fileInput?.click();
});

// Keyboard: Enter/Space on drop zone
dropZone?.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); }
});

// File input change
fileInput?.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) uploadFile(file);
  fileInput.value = '';  // allow re-selecting same file
});

// Drag and drop
dropZone?.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone?.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone?.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer?.files?.[0];
  if (file) uploadFile(file);
});

// "Change" button — show drop zone again
btnChange?.addEventListener('click', () => {
  previewArea?.classList.add('hidden');
  dropZone?.classList.remove('hidden');
  fileInput?.click();
});

// Export init function for main.js to call
export function initReference() {
  // Nothing extra to initialise — event listeners are registered above.
  // main.js imports this module to ensure it's loaded.
}
