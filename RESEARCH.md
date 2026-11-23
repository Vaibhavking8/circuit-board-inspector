# RESEARCH.md

What I looked into before choosing the approach for BoardInspect.

> Status: updated 20 Nov 2026.  
> **[EXPERIMENT]** = claim not yet validated on real or synthetic test images.  
> **[VERIFY]** = citation I have not yet opened and confirmed myself.

---

## 1. Problem restated

A fixed camera photographs the same board every time. The user names parts once on one good photo (the reference). For each new photo, report which named parts are missing and where, or show a green **OK**.

This is **presence/absence checking at known, user-defined locations** — not open-ended defect detection. That shapes every choice below.

---

## 2. Image alignment

### 2.1 Transformation families

| Model | DOF | Matrix | Physical meaning |
|---|---|---|---|
| Translation | 2 | 2×3 | Shift only |
| Euclidean (rigid) | 3 | 2×3 | Rotation + translation; distances preserved |
| Similarity | 4 | 2×3 | + uniform scale |
| Affine | 6 | 2×3 | Parallel lines preserved |
| Homography (projective) | 8 | 3×3 | Straight lines preserved; models perspective |

Sources: OpenCV geometric transformations tutorial (docs.opencv.org/4.x, `tutorial_py_geometric_transformations.html`).

### 2.2 Chosen strategy: ECC cascade (Euclidean → Affine)

**Why ECC first:**  ECC (`cv2.findTransformECC`) is an area-based, intensity-driven alignment method that maximises the Enhanced Correlation Coefficient between two images. Unlike feature-based methods it does not need identifiable keypoints, so it works on textureless or repetitively textured boards. It is documented as robust to photometric changes (brightness, contrast). **[VERIFY against Evangelidis & Psarakis TPAMI 2008]**

**Motion model selection:** A board lying flat on a conveyor and photographed from directly above needs at most a Euclidean transform (3 DOF). Affine (6 DOF) handles mild shear or lens distortion. Homography (8 DOF) can over-fit on a board with repetitive texture, distorting a part's appearance and triggering false alarms. **[EXPERIMENT]**

**Cascade rationale:** try ECC Euclidean first; if the returned ECC quality score is below `config.ECC_MIN_QUALITY`, retry with ECC affine. If that also fails, fall back to ORB + `estimateAffinePartial2D` as an initial-guess supplier, then run ECC affine on top of it. Use homography only if an experiment proves perspective tilt is real on the target camera.

**Convergence parameters (config defaults):**

| Parameter | Default | Why |
|---|---|---|
| `ECC_MAX_ITER` | 100 | Covers typical small displacements without excessive runtime |
| `ECC_EPS` | 1e-5 | Finer than the 1e-3 OpenCV default; trades a few ms for sub-pixel accuracy **[EXPERIMENT]** |
| `ECC_GAUSS_SIZE` | 5 | Gaussian smoothing reduces noise sensitivity during optimisation **[VERIFY]** |
| `REG_RESOLUTION` | 1200 px (long edge) | See section 6 |

**Transform validation safeguards:**  
- Reject if the determinant of the rotation block deviates from 1.0 by more than `config.MAX_DET_DEVIATION` (default 0.05). A very different value means the solver found a degenerate warp.  
- Reject if the recovered rotation angle exceeds `config.MAX_ROTATION_DEG` (default 10°). Larger rotations suggest ECC diverged.  
- Reject if the recovered translation exceeds `config.MAX_TRANSLATION_PX` (default 80 px at registration resolution). Both checks raise `RegistrationError`, which the API turns into a user-visible message.

### 2.3 ECC vs ORB comparison

| Approach | Handles big shifts | Sub-pixel accuracy | Needs texture | Failure mode | Role in pipeline |
|---|---|---|---|---|---|
| ECC Euclidean | Small–moderate | Yes | No | Non-convergence exception | **Primary** |
| ECC Affine | Small–moderate | Yes | No | Over-fitting at 6 DOF | Secondary fallback |
| ORB + `estimateAffinePartial2D` | Larger | Moderate | Yes (keypoints) | Bad matches on repetitive board **[EXPERIMENT]** | Initial-guess supplier only |
| ECC Homography | Small–moderate | Yes | No | 8 DOF over-fits, distorts parts | Rejected unless tilt proven **[EXPERIMENT]** |

**[EXPERIMENT]** Planned: warp one reference with known shifts 0–20 px, rotations 0–5°, scale 0.98–1.02, brightness ±30 %, Gaussian noise σ 5. Measure per-method mean corner error and false-alarm rate on a complete board. Record results in a table in `docs/`.

---

## 3. Part comparison signals

### 3.1 Template match (`TM_CCOEFF_NORMED`)

After alignment, each named part has a bounding box from the reference. `cv2.matchTemplate` with `TM_CCOEFF_NORMED` slides the reference crop over a **padded search window** in the test image and returns the peak score. The window absorbs a few pixels of residual alignment error.

`TM_CCOEFF_NORMED` subtracts each patch's mean before correlating, making it less sensitive to overall brightness shifts than raw `TM_CCORR`. **[EXPERIMENT: compare false-alarm rates across methods on a brightness-shifted complete board]**

Only `TM_SQDIFF` and `TM_CCORR_NORMED` accept masks; for `TM_CCOEFF_NORMED` an alpha-masked crop is not needed because the padded-window approach already confines the search area.

### 3.2 Edge/gradient signal (illumination invariance)

A missing component removes strong edges (component body, pins, shadow boundary). The edge signal — computed with `cv2.Sobel` or `cv2.Canny` on the cropped region — is compared between reference and test crops via correlation or simple intersection. This signal is largely invariant to overall brightness changes because edge detection operates on gradients, not absolute intensities. **[EXPERIMENT: measure precision/recall contribution relative to template score alone]**

The combined score is a weighted sum `config.TEMPLATE_WEIGHT * template_score + config.EDGE_WEIGHT * edge_score`.

### 3.3 Why SSIM is not included by default

SSIM (Structural Similarity Index, Wang et al. 2004) measures luminance, contrast and structure. It is used widely in image quality research. However:

- It requires `scikit-image`, an extra dependency.
- On small part crops (10–50 px), the multi-scale variant is not meaningful and the single-scale version is dominated by mean luminance, which the template match already captures.
- **[EXPERIMENT]** A pilot run on the synthetic test set showed no measurable improvement in F1 when SSIM was added as a third signal alongside template + edge. If a broader test dataset reverses this finding, SSIM will be added behind `config.USE_SSIM = False`.

---

## 4. Decision engine

### 4.1 Z-score margin and threshold placement

Each part is calibrated by running augmented copies of the reference through the same comparison pipeline and collecting score samples. Augmentations include small shifts (±3 px), brightness changes (±20 %), Gaussian noise (σ ≤ 5) and mild blur.

The **present distribution** mean is `μ_present` and standard deviation `σ_present`. The **absent proxy** distribution (inpainting the reference crop to simulate a missing part — see section 5) gives `μ_absent`.

The threshold is placed at the midpoint: `τ = (μ_present + μ_absent) / 2`. The sensitivity slider shifts `τ` by ±`k × σ_present`, where `k` is `config.SENSITIVITY_K` (default 0.5). An **uncertainty band** `[τ − config.BAND_HALF, τ + config.BAND_HALF]` is marked on the UI score bar; scores inside the band are shown as amber.

### 4.2 Why this is conservative

Inpainting produces the most board-like background that can be generated without real absent-part photos, but it is not guaranteed to match a true missing-part photo. The midpoint threshold therefore errs toward reporting present if scores are ambiguous. The user can lower sensitivity if they see false alarms. Per-part calibration is re-run if the reference image or any box is changed.

---

## 5. Calibration: synthetic augmentation and inpainting

The only "data" for threshold calibration is the single reference photo. The approach:

1. **Present samples** — augment the reference crop (shift, brightness, noise, blur). These are the scores a good board produces.
2. **Absent proxy** — inpaint the reference crop using `cv2.inpaint` (Telea or Navier-Stokes), then apply the same augmentations. The result is the board PCB substrate with the component removed.

The inpainting proxy is conservative: it does not capture the exact look of an absent component on a real board, so the absent distribution and present distribution may overlap. Per-part thresholds will be wider (less sensitive) than optimal. **[EXPERIMENT: compare inpaint proxy to real absent images if any become available]**

---

## 6. Working resolutions

| Step | Resolution (long edge) | Rationale |
|---|---|---|
| Registration | 1200 px | Enough for ECC to converge on board features; faster than full-res **[EXPERIMENT]** |
| Comparison | 2400 px | Preserves sub-part detail on small components **[EXPERIMENT]** |
| Storage | Original | Normalised coordinates are resolution-independent; never re-annotate |

**[EXPERIMENT]** Planned: sweep resolution from 400 to full-res and measure alignment error and runtime. Record break-even point.

---

## 4.4 Why not Streamlit (Decision 7)

The UI is part of the evaluation score. The core interaction — drawing a box around each part and naming it — requires precise control over canvas coordinate handling, hit-testing, inline label editing and overlay rendering. This drove the framework choice.

| | Streamlit + drawable canvas | FastAPI + plain HTML/JS canvas |
|---|---|---|
| Box-draw-and-name flow | Built around a third-party canvas component; name editing requires workarounds | Fully custom; exact UX the user needs |
| Look and feel | Streamlit's default theme; limited styling | Full CSS control over layout, colours, result cards, overlays |
| Background image stability | v0.10.0 release notes document that `background_image` broke on newer Streamlit versions | No dependency; browser `<canvas>` drawImage is stable |
| Coordinate mapping | Component handles scaling with limited control | One module (`coordinate-utils.js`) with explicit, tested transforms |
| Offline telemetry | Streamlit sends usage statistics by default; needs opt-out | No outbound connections |
| Demo risk | Third-party component version mismatch on a fresh laptop | Only standard browser APIs |

Sources: streamlit-drawable-canvas v0.10.0 release notes (https://github.com/andfanilo/streamlit-drawable-canvas/releases/tag/v0.10.0); Streamlit forum thread on the component (https://discuss.streamlit.io/t/drawable-canvas/3671).

**Decision (from AI_COLLAB.md Decision 7):** FastAPI + plain HTML/JS. The draw-a-box interaction is the centrepiece of the UI score, so full control matters more than speed of setup. Streamlit was considered and rejected on the grounds of coordinate-mapping fragility and demo risk. This is recorded as a final decision by the project owner.

---

## 7. Dependencies justified

| Library | Pin | Justification |
|---|---|---|
| `opencv-python-headless` | ≥4.8 | All CV operations: alignment, template match, edge, inpaint, draw |
| `numpy` | ≥1.24 | Array operations throughout the CV stack |
| `fastapi` | ≥0.110 | Async backend, automatic JSON schema validation, test client |
| `uvicorn[standard]` | ≥0.29 | ASGI server for FastAPI |
| `python-multipart` | ≥0.0.9 | Required for file-upload form parsing in FastAPI |
| `pytest` | dev | Unit and API integration tests |
| `httpx` | dev | FastAPI `TestClient` transport |
| `scikit-image` | optional | SSIM only if `config.USE_SSIM = True`; excluded from default install |
| `Pillow` | excluded | OpenCV handles all image I/O; add only if a format proves awkward |
| `Plotly` / `pandas` | excluded | No charts needed; a result card list is sufficient |
| `SQLite` | excluded | JSON files are sufficient for reference-setup storage at this scale |

---

## 8. OpenCV documentation used

- Geometric transformations tutorial: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html  
- `findTransformECC` reference (OpenCV 4.13.0 video tracking group): https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html  
- Template matching tutorial: https://docs.opencv.org/4.13.0/de/da9/tutorial_template_matching.html  
- `estimateAffinePartial2D` and calib3d functions: https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html  
- Evangelidis & Psarakis ECC paper (cited in OpenCV docs): *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization*, IEEE TPAMI 2008. **[VERIFY link]**

---

## 9. Open questions

1. How much does real perspective tilt matter for this camera setup? (decides Euclidean/affine vs homography)
2. Is an ORB initial guess worth the extra code, or does identity converge reliably for typical board shots?
3. Do edge signals add measurable F1 on top of template matching once thresholds are per-part?
4. Does the inpainting absent-proxy overestimate absent scores, and by how much?
5. What is the break-even resolution where ECC accuracy stops improving and runtime dominates?
