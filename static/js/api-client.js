/**
 * api-client.js — uniform fetch wrapper for the BoardInspect API.
 *
 * All HTTP calls go through request() or requestFile(). Non-2xx responses
 * throw an Error with the server's error message. Network failures are caught
 * and re-thrown with a friendly message.
 * @module api-client
 */

const BASE = '';  // same origin — server and frontend are co-located

/**
 * Make a JSON API request.
 * @param {'GET'|'POST'|'PUT'|'DELETE'} method
 * @param {string} path   e.g. '/api/reference'
 * @param {object|null} [body]  JSON body (omit for GET)
 * @returns {Promise<any>}  Parsed JSON response
 */
export async function request(method, path, body = null) {
  const opts = {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  };
  let resp;
  try {
    resp = await fetch(BASE + path, opts);
  } catch (err) {
    throw new Error(`Network error: ${err.message}`);
  }
  if (!resp.ok) {
    let msg = `Server error ${resp.status}`;
    try {
      const data = await resp.json();
      msg = data.error || data.detail || msg;
    } catch { /* ignore parse error */ }
    throw new Error(msg);
  }
  return resp.json();
}

/**
 * Upload a file (multipart/form-data).
 * @param {string} path
 * @param {File|Blob} file
 * @param {string} [fieldName='file']
 * @param {object} [extraParams]  Additional URL query params
 * @returns {Promise<any>}
 */
export async function requestFile(path, file, fieldName = 'file', extraParams = {}) {
  const form = new FormData();
  form.append(fieldName, file);
  const url = new URL(BASE + path, location.href);
  for (const [k, v] of Object.entries(extraParams)) url.searchParams.set(k, v);
  let resp;
  try {
    resp = await fetch(url, { method: 'POST', body: form });
  } catch (err) {
    throw new Error(`Network error: ${err.message}`);
  }
  if (!resp.ok) {
    let msg = `Server error ${resp.status}`;
    try {
      const data = await resp.json();
      msg = data.error || data.detail || msg;
    } catch { /* ignore */ }
    throw new Error(msg);
  }
  return resp.json();
}

/**
 * Poll a job until complete or failed.
 * @param {string} jobId
 * @param {function} onProgress   Called with (progress: 0-1) on each poll
 * @param {number}   [intervalMs=800]
 * @returns {Promise<object>}  The completed job result
 */
export async function pollJob(jobId, onProgress, intervalMs = 800) {
  for (;;) {
    const job = await request('GET', `/api/jobs/${jobId}`);
    onProgress(job.progress ?? 0);
    if (job.status === 'complete') return job;
    if (job.status === 'failed') throw new Error(job.error || 'Job failed');
    await new Promise(r => setTimeout(r, intervalMs));
  }
}
