/**
 * state.js — single source of truth for BoardInspect UI state.
 *
 * All mutable UI state lives here. Other modules read via getters and
 * mutate via setters. Setters notify subscribers so the UI re-renders.
 * @module state
 */

/** @typedef {{ id: string, name: string, box: [number,number,number,number] }} Part */

const _state = {
  referenceId: null,         // string | null
  referenceSize: null,       // { width, height } | null
  parts: [],                 // Part[]
  currentStep: 1,            // 1 | 2 | 3
  sensitivity: 50,           // 0-100 (maps to 0.0-1.0)
  calibrationStatus: null,   // null | 'running' | 'complete' | 'failed'
  results: [],               // ImageResultResponse[]
  jobIds: [],                // string[] — active job ids
};

/** @type {Map<string, Set<Function>>} */
const _subscribers = new Map();

/**
 * Subscribe to state changes for a given key.
 * @param {string} key
 * @param {Function} fn  Called with (newValue, oldValue) on change.
 */
export function subscribe(key, fn) {
  if (!_subscribers.has(key)) _subscribers.set(key, new Set());
  _subscribers.get(key).add(fn);
}

function _notify(key, newVal, oldVal) {
  _subscribers.get(key)?.forEach(fn => fn(newVal, oldVal));
}

function _set(key, value) {
  const old = _state[key];
  _state[key] = value;
  _notify(key, value, old);
}

// ── Getters ─────────────────────────────────────────────────────────────────
export const getReferenceId       = () => _state.referenceId;
export const getReferenceSize     = () => _state.referenceSize;
export const getParts             = () => [..._state.parts];
export const getCurrentStep       = () => _state.currentStep;
export const getSensitivity       = () => _state.sensitivity;
export const getCalibrationStatus = () => _state.calibrationStatus;
export const getResults           = () => [..._state.results];
export const getJobIds            = () => [..._state.jobIds];

// ── Setters ─────────────────────────────────────────────────────────────────
export function setReferenceId(id)      { _set('referenceId', id); }
export function setReferenceSize(size)  { _set('referenceSize', size); }
export function setParts(parts)         { _set('parts', [...parts]); }
export function setCurrentStep(step)    { _set('currentStep', step); }
export function setSensitivity(val)     { _set('sensitivity', val); }
export function setCalibrationStatus(s) { _set('calibrationStatus', s); }
export function setResults(results)     { _set('results', [...results]); }
export function addResult(result)       { _set('results', [..._state.results, result]); }
export function addJobId(jid)           { _set('jobIds', [..._state.jobIds, jid]); }
export function clearJobIds()           { _set('jobIds', []); }

/** Reset all state (Start over). */
export function reset() {
  _set('referenceId', null);
  _set('referenceSize', null);
  _set('parts', []);
  _set('currentStep', 1);
  _set('sensitivity', 50);
  _set('calibrationStatus', null);
  _set('results', []);
  _set('jobIds', []);
}
