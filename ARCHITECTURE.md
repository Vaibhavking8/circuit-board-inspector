# ARCHITECTURE.md

System architecture for BoardInspect — an offline desktop web app for PCB missing-component detection.

> Last updated: 20 Nov 2026

---

## 1. System overview

BoardInspect runs entirely on the user's local machine. The backend is a FastAPI/Uvicorn process bound to `127.0.0.1` only. The browser loads one HTML page served from that same process, then communicates with it over JSON/HTTP. No external hosts are contacted at any point during normal operation. All CV work (alignment, template matching, edge comparison, decision) happens in pure Python packages that have no dependency on the web framework. The frontend is plain HTML, CSS and ES modules — no build step, no JavaScript framework, no CDN assets.

---

## 2. Layer diagram

```
┌─────────────────────────────────────┐
│  Browser (HTML/CSS/JS ES modules)   │
│  index.html · annotation-canvas.js  │
│  results.js · coordinate-utils.js   │
└──────────────────┬──────────────────┘
                   │ HTTP JSON (loopback only, 127.0.0.1)
┌──────────────────▼──────────────────┐
│  FastAPI + Uvicorn  (api/)          │
│  routes · schemas · jobs · services │
└──────────────────┬──────────────────┘
                   │ Python calls
┌──────────────────▼──────────────────┐
│  Pipeline (pipeline.py)             │
│  orchestrates the CV packages       │
└──┬────────┬──────────┬──────────────┘
   │        │          │
imaging  registration  comparison + decision
   │        │          │
 (OpenCV / NumPy — pure, no web framework)
              │
           storage.py  (JSON + PNG on local disk)
```

The `api/services/` layer is a thin adapter: it converts HTTP request data into domain objects, calls the pure packages, and converts results back to JSON schemas. No CV logic belongs there.

---

## 3. Key design decisions and trade-offs

| Decision | Choice | Rejected alternative | Reason |
|---|---|---|---|
| **Backend framework** | FastAPI | Flask | FastAPI provides automatic Pydantic schema validation, async request handling, and a built-in test client (`TestClient` via `httpx`). Flask is simpler but requires manual validation. The difference matters for the job-queue pattern used for long-running inspections. |
| **Frontend** | Plain HTML/CSS/ES modules | Streamlit | The draw-a-box-and-name-it interaction is the centrepiece of the UI score and requires precise canvas coordinate handling, inline label editing and hit-testing. Streamlit's drawable canvas is a third-party component whose `background_image` feature broke across Streamlit versions; on a fresh demo laptop that is a live risk. Full HTML/JS gives complete control over look, layout and coordinates. See RESEARCH.md section 4.4 and AI_COLLAB.md Decision 7. |
| **Registration order: ECC before ORB** | ECC (Euclidean → Affine) first | ORB + RANSAC first | ECC is more accurate for the small displacements expected from a fixed camera (sub-pixel, no keypoint noise). ORB handles larger warps but can produce bad matches on repetitive board texture. ORB is used only as an initial-guess supplier when ECC fails to converge. |
| **Coordinate storage: normalised** | `x, y, w, h` in [0, 1] relative to original image size | Pixel coordinates | Normalised coordinates are resolution-independent. If the user photographs at a different resolution they never need to re-annotate. One canonical Python converter and a mirrored JS module prevent drift; both are tested against shared vectors. |
| **Threshold calibration: synthetic** | Augment the reference; use inpainting as absent proxy | Labelled absent-part photos | No labelled data is available. Augmentation is deterministic from one reference photo. The inpainting proxy is conservative (overlapping distributions → wider threshold → lower false-negative risk at the cost of some false-positive risk). |
| **Network binding: loopback only** | `127.0.0.1` | `0.0.0.0` | The demo runs on a personal laptop in an interview setting. Binding to all interfaces would expose the service on whatever networks the laptop is connected to. Loopback eliminates that exposure. CORS is disabled; the browser and server share the same origin. |
| **JavaScript libraries** | None (vanilla ES modules) | React, Vue, canvas toolkits | No build step, no npm, no CDN. The only JS code is what I wrote. If a library becomes truly necessary it is vendored into `static/vendor/` with its licence. |
| **Storage format** | JSON + PNG files | SQLite | At the scale of a few reference setups on one machine, flat files are simpler to debug, back up and inspect. The JSON schema carries a version field. A content hash in the setup record detects accidental file corruption. |

---

## 4. Inspection data flow

```
Browser upload (multipart/form-data)
  │
  ▼
routes.py  — validate file size, extension, multipart integrity
  │
  ▼
imaging/   — decode image bytes with cv2.imdecode
           — check dimensions (min/max px)
           — EXIF orientation normalisation
  │
  ▼
registration/  — load reference image at REG_RESOLUTION
              — ECC Euclidean; on failure ECC Affine; on failure ORB initial guess + ECC Affine
              — validate transform (det, angle, translation bounds)
              — warp test image into reference frame
  │
  ▼
comparison/  — for each part box:
             —   extract padded search window from warped test image
             —   cv2.matchTemplate (TM_CCOEFF_NORMED) → template score
             —   Sobel edges on reference crop and test crop → edge score
             —   weighted sum → combined score
  │
  ▼
decision/  — look up per-part calibrated threshold
           — score > τ + band/2  → PRESENT
           — score < τ − band/2  → MISSING
           — otherwise           → UNCERTAIN (shown as amber in UI)
  │
  ▼
routes.py  — assemble ImageResult (list of PartResult + overall verdict)
           — return JSON response
  │
  ▼
Browser   — results.js renders result cards and annotated canvas overlay
```

Long-running inspections (multiple images) are queued via `jobs.py` and polled by the browser.

---

## 5. Storage format

Each saved reference setup lives in `data/runtime/setups/<id>/`:

```
setup.json   — schema_version, setup_id, content_hash (SHA-256 of reference image bytes),
               reference_image_path, original_width, original_height,
               parts: [ { id, name, x, y, w, h } ]   ← normalised [0,1]
<id>.png     — reference image at original resolution
```

`schema_version` is bumped on any breaking change to the JSON layout. The API rejects setups whose version it does not understand and returns a clear error. `content_hash` is verified on load; a mismatch raises `CorruptSetupError`.

---

## 6. Module dependency rules

```
api/          may import anything below it
pipeline.py   may import imaging, registration, comparison, decision, config, models
imaging/      may import config, models, errors       (no api, no storage)
registration/ may import config, models, errors       (no api, no storage)
comparison/   may import config, models, errors       (no api, no storage)
decision/     may import config, models, errors       (no api, no storage)
storage.py    may import config, models, errors       (no api, no CV packages)
tools/        may import src/boardinspect             (never the reverse)
```

Algorithm packages must not import the web framework or perform file I/O. All I/O goes through `imaging/` (decode) and `storage.py` (persist). This keeps the CV code testable without starting the server.
