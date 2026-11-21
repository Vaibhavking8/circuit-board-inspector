/**
 * main.js — entry point. Wires modules, drives the stepper, handles start-up.
 * @module main
 */

import {
  getCurrentStep, setCurrentStep,
  getReferenceId, getParts,
  getCalibrationStatus, reset,
  subscribe,
} from './state.js';
import { request } from './api-client.js';

// ── DOM references ──────────────────────────────────────────────────────────
const stepBtns   = [1, 2, 3].map(n => document.getElementById(`step-btn-${n}`));
const sections   = ['step-reference', 'step-annotate', 'step-inspect']
                     .map(id => document.getElementById(id));
const btnNext    = document.getElementById('btn-next-to-annotate');
const btnCalib   = document.getElementById('btn-calibrate');
const btnBackRef = document.getElementById('btn-back-to-reference');
const btnBackAnn = document.getElementById('btn-back-to-annotate');
const btnStart   = document.getElementById('btn-start-over');
const sensSlider = document.getElementById('sensitivity-slider');
const sensValue  = document.getElementById('sensitivity-value');
const btnDebug   = document.getElementById('btn-toggle-debug');
const debugPanel = document.getElementById('debug-content');

// ── Stepper rendering ───────────────────────────────────────────────────────
function renderStepper(step) {
  stepBtns.forEach((btn, i) => {
    const n = i + 1;
    btn.classList.toggle('active', n === step);
    btn.classList.toggle('done',   n < step);
    // Only allow clicking completed steps to go back
    btn.disabled = n > step;
    btn.setAttribute('aria-current', n === step ? 'step' : 'false');
  });
  sections.forEach((sec, i) => {
    const active = (i + 1) === step;
    sec.classList.toggle('active', active);
    sec.classList.toggle('hidden', !active);
  });
}

function goToStep(n) {
  setCurrentStep(n);
  renderStepper(n);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Allow clicking done step buttons to go back
stepBtns.forEach((btn, i) => {
  btn.addEventListener('click', () => {
    const n = i + 1;
    if (n < getCurrentStep()) goToStep(n);
  });
});

// ── Step 1 → 2 ──────────────────────────────────────────────────────────────
btnNext?.addEventListener('click', () => {
  if (getReferenceId()) goToStep(2);
});

// Enable "Next" when referenceId is set
subscribe('referenceId', id => {
  if (btnNext) btnNext.disabled = !id;
});

// ── Step 2 → 3 (calibrate) ──────────────────────────────────────────────────
btnCalib?.addEventListener('click', () => {
  if (getParts().length > 0 && getReferenceId()) goToStep(3);
});

// Enable "Calibrate" when parts exist
subscribe('parts', parts => {
  if (btnCalib) btnCalib.disabled = parts.length === 0;
});

// ── Back buttons ─────────────────────────────────────────────────────────────
btnBackRef?.addEventListener('click', () => goToStep(1));
btnBackAnn?.addEventListener('click', () => goToStep(2));

// ── Start over ───────────────────────────────────────────────────────────────
btnStart?.addEventListener('click', () => {
  if (confirm('Start over? All current work will be lost.')) {
    const refId = getReferenceId();
    if (refId) {
      request('POST', `/api/reference/${refId}/reset`).catch(() => {});
    }
    reset();
    goToStep(1);
    // Reload to clear all module state cleanly
    location.reload();
  }
});

// ── Sensitivity slider ───────────────────────────────────────────────────────
sensSlider?.addEventListener('input', () => {
  const val = sensSlider.value;
  if (sensValue) sensValue.textContent = val;
  sensSlider.setAttribute('aria-valuenow', val);
});

// ── Debug toggle ─────────────────────────────────────────────────────────────
btnDebug?.addEventListener('click', () => {
  const hidden = debugPanel.classList.toggle('hidden');
  btnDebug.textContent = hidden ? '▶ Show per-part scores' : '▼ Hide per-part scores';
});

// ── Health check on start-up ─────────────────────────────────────────────────
async function init() {
  try {
    await request('GET', '/health');
    console.log('BoardInspect server: OK');
  } catch (err) {
    console.warn('Health check failed:', err.message);
  }
  renderStepper(getCurrentStep());
}

init();
