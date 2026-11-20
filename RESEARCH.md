# RESEARCH.md

What I looked into before choosing the approach for the Missing Part Detector.

> Status: draft, 19 Nov 2026. Items marked **[VERIFY]** are citations I have not yet opened and checked myself. Items marked **[EXPERIMENT]** are claims I still need to confirm on my own test images before relying on them.

---

## 1. Problem restated

- A fixed camera photographs the same board every time.
- The user names parts once on one good photo (the reference).
- For each new photo, report which named parts are missing and where, or show a green **OK**.
- No training data is provided, no internet at runtime, no hosted APIs.

This is **presence/absence checking at known locations**, not open-ended defect detection. That shapes every choice below.

---

## 2. Image alignment: the transformation concepts

Even with a "fixed" camera, a board can sit a few pixels off, be slightly rotated on a conveyor, or the photo can be a bit tilted. If the new photo is not aligned with the reference, the box around "C1" lands on the wrong pixels and a good board looks like it has a missing part. So I need to understand which kind of transformation to correct for.

### 2.1 The transformation families

Each family is a bigger set of allowed movements than the one before it.

| Model | What it allows | Degrees of freedom (DOF) | Matrix | Key property |
|---|---|---|---|---|
| **Translation** | Shift in x and y | 2 | 2×3 | Nothing rotates or scales |
| **Euclidean (rigid)** | Rotation + translation | 3 | 2×3 | Preserves distances and angles. A board sliding and turning on a flat conveyor. |
| **Similarity** | Rotation + translation + **uniform** scale | 4 | 2×3 | Preserves angles and shape. Camera moved slightly closer or farther. |
| **Affine** | Rotation, translation, scale (can differ in x and y), shear | 6 | 2×3 | **Parallel lines stay parallel.** |
| **Projective (homography)** | All of affine, plus perspective | 8 | 3×3 | **Straight lines stay straight, but parallel lines may converge.** Models a flat surface seen from a different camera angle. |

Sources for these definitions: OpenCV's geometric transformations tutorial states that in an affine transformation all parallel lines stay parallel, and that a perspective transformation needs a 3×3 matrix and keeps straight lines straight. OpenCV's ECC documentation lists the Euclidean model as 3 estimated parameters (2×3 matrix), affine as 6 (2×3), and homography as 8 (3×3). The partial affine function is documented as 4 DOF, limited to translation, rotation and uniform scaling.

Matrix forms (standard):

```
Translation / Euclidean / Similarity / Affine  ->  2x3 matrix, applied with cv2.warpAffine
    [ a  b  tx ]
    [ c  d  ty ]

Homography  ->  3x3 matrix, applied with cv2.warpPerspective
    [ h11 h12 h13 ]
    [ h21 h22 h23 ]
    [ h31 h32  1  ]     (8 free numbers, bottom-right fixed to 1)
```

### 2.2 Which one fits this problem?

The key physical question: **what actually changes between the reference photo and a test photo?**

- Board slides or rotates **flat on the same plane** (camera looks straight down): Euclidean is the exact model, 3 DOF.
- Slight zoom change (camera height or lens drift): add uniform scale, which is a similarity.
- Lens/sensor quirks or a tilted board: affine or homography.
- Camera genuinely at a different angle to a flat board: homography is the physically correct model.

Fewer DOF is **safer**: with only 3 numbers to estimate, there is much less room for the solver to find a wrong answer on a board with repetitive texture. A homography has 8 free numbers and can bend the image in ways that distort a part's appearance and cause false alarms. **[EXPERIMENT]**

**Working choice (provisional):** try **ECC with Euclidean**, fall back to **ECC affine** if the residual error is high. Use homography only if tests show perspective tilt is real. I will decide with measurements (see section 2.5), not by guessing.

### 2.3 Two ways to estimate the transform

#### A) Intensity-based: ECC (`cv2.findTransformECC`)

- Aligns two images by maximising the **enhanced correlation coefficient** between them (area-based, uses pixel intensities directly, no feature detection).
- Supports `MOTION_TRANSLATION`, `MOTION_EUCLIDEAN`, `MOTION_AFFINE` (the default) and `MOTION_HOMOGRAPHY`.
- The OpenCV documentation notes that unlike `findHomography` and `estimateRigidTransform`, ECC is an area-based alignment that builds on intensity similarities. It refines an **initial** transformation; if none is given it starts from identity. With strong displacement or rotation you need a rough initial guess first.
- It **throws an exception if it does not converge**, so the code must catch that and handle it (for example, report "could not align this image" rather than crashing).
- **Verified against the official OpenCV 4.13.0 reference** (https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html): `findTransformECC` takes `templateImage`, `inputImage`, `warpMatrix`, `motionType` (default `MOTION_AFFINE`), `criteria` (default 50 iterations or an increment below 0.001) and an optional `inputMask`. A separate `findTransformECCWithMask` takes validity masks for **both** the template and the input image (its defaults are 50 iterations, 1e-6 and a Gaussian filter size of 5), which could be used to exclude the user-boxed parts from alignment **[EXPERIMENT]**. `computeECC` returns the ECC value between two images and may serve as an alignment quality score **[EXPERIMENT]**. The translation and Euclidean enumerators appear on that page; the affine and homography enumerators are listed in the `ECCParameters` documentation linked below.
- ECC is designed to be robust to photometric changes (brightness and contrast), which suits a factory with changing light. **[VERIFY against the paper]**
- Practical note: ECC finds the warp that maps the template to the input, so to bring the *test* image into the reference frame use `cv2.warpAffine(..., flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)`. **[EXPERIMENT: confirm direction with a known shift]**
- Multi-scale variant: `findTransformECCMultiScale` and the `ECCParameters` struct (pyramid levels, Gaussian filter size) are documented in OpenCV 5.

**Pros:** sub-pixel accurate, works on textureless boards where feature detectors fail, handles small motions very well.
**Cons:** needs a decent starting point, can fail to converge, slower on large images (mitigate by running at reduced resolution).

#### B) Feature-based: ORB + RANSAC (`estimateAffinePartial2D`, `estimateAffine2D`, `findHomography`)

1. Detect keypoints and descriptors (ORB).
2. Match descriptors between reference and test.
3. Fit a transform to the matched points with **RANSAC**, which ignores wrong matches.

| Function | Model | DOF |
|---|---|---|
| `cv2.estimateAffinePartial2D` | translation + rotation + uniform scale | 4 |
| `cv2.estimateAffine2D` | full affine | 6 |
| `cv2.findHomography` | projective | 8 |
| `cv2.getAffineTransform` | exact affine from **3** point pairs (no outlier handling) | 6 |
| `cv2.getPerspectiveTransform` | exact homography from **4** point pairs (no outlier handling) | 8 |

Notes:
- `estimateAffinePartial2D` is documented as "an optimal limited affine transformation with 4 degrees of freedom", with RANSAC as the default method.
- It includes **uniform scale**. A forum thread on OpenCV confirms people looking for a pure rigid (no scale) transform notice this. For strictly rigid alignment, ECC Euclidean is the cleaner OpenCV option.
- `getAffineTransform` and `getPerspectiveTransform` take exactly 3 and 4 point pairs and have no robustness to bad matches, so they suit hand-picked points, not automatic matching.

**Pros:** handles larger displacement without an initial guess.
**Cons:** needs enough distinctive features. A mostly green board with repeated, similar-looking parts can produce wrong matches. **[EXPERIMENT]**

### 2.4 Comparison summary

| Approach | Handles big shifts | Sub-pixel accuracy | Needs texture | Failure mode | My verdict |
|---|---|---|---|---|---|
| ECC Euclidean | Small to moderate | Yes | No | Non-convergence exception | **First choice** |
| ECC affine | Small to moderate | Yes | No | Non-convergence, slight over-fitting | Fallback |
| ECC homography | Small to moderate | Yes | No | Over-fitting, 8 parameters | Only if tilt is proven |
| ORB + `estimateAffinePartial2D` | Larger | Moderate | Yes | Bad matches on repetitive board | Initial guess / fallback |
| ORB + `findHomography` | Larger | Moderate | Yes | Distortion from wrong matches | Rejected unless tilt is proven |

### 2.5 Experiment plan (to turn the above into evidence)

Use one reference image and synthetically warp it with **known** transforms, then measure how well each method recovers them.

- [ ] Shifts of 0 to 20 px, rotations of 0 to 5 degrees, scale 0.98 to 1.02, small perspective tilt.
- [ ] For each of ECC translation/Euclidean/affine/homography and ORB + partial affine: record alignment error (mean corner error in px), runtime, and convergence failures.
- [ ] Count **false "missing" alerts on complete boards** under each method. This is the real metric.
- [ ] Add brightness changes, noise and mild blur on top.
- [ ] Record results in a table here and link the notebook or script.

---

## 3. Comparing a part to its reference: template matching

After alignment, each named part has a box from the reference. I compare that reference crop with the test image.

- `cv2.matchTemplate` slides the template over the image and computes a similarity score at each position.
- OpenCV documents six methods: `TM_SQDIFF`, `TM_SQDIFF_NORMED`, `TM_CCORR`, `TM_CCORR_NORMED`, `TM_CCOEFF`, `TM_CCOEFF_NORMED`. For the two SQDIFF methods the best match is the **lowest** value; for the rest, the **highest**.
- `TM_CCOEFF_NORMED` subtracts the mean of the template and of the image patch before correlating, which makes it less sensitive to overall brightness. This is why I prefer it for lighting changes. **[EXPERIMENT]**
- Only `TM_SQDIFF` and `TM_CCORR_NORMED` accept a mask in current OpenCV docs.
- A community tutorial notes that raw `TM_CCORR` gave poor results compared with the other methods in its example, which is another reason to avoid un-normalised variants.

**Key design idea:** search inside a **padded window** around each box instead of comparing one fixed rectangle, and take the **best score** in the window. This absorbs a few pixels of leftover alignment error. A part counts as present if the best score is above that part's calibrated threshold.

Extra signals to consider only if the test data shows they are needed: edge similarity (missing parts remove strong edges), SSIM. **[EXPERIMENT]**

---

## 4. Existing approaches and data

### 4.1 Approaches considered

| Option | Idea | Decision | Why |
|---|---|---|---|
| **Reference-based classical CV** (alignment + template/edge comparison) | Compare each user-named region with the reference | **Chosen** | Works from a single reference photo, no training, explainable, runs offline on a laptop |
| YOLO or other supervised detectors | Train a model to find components | Rejected | Needs many labelled images; the user names parts live from one photo, so there is nothing to train on |
| Anomaly detection (for example PatchCore) | Learn "normal" appearance and flag deviations | Rejected | Heavier to set up, harder to explain, and it flags *any* difference rather than the specific named parts |
| Full-image differencing only | Subtract reference from test | Rejected as the sole method | Fragile to lighting, small shifts and noise; kept only as an extra signal |

PatchCore reference: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, 2021 (arXiv). **[VERIFY link]**

### 4.2 Academic and library references

- OpenCV ECC (`findTransformECC`) is based on Evangelidis and Psarakis, *Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization*, IEEE TPAMI, 2008 (cited as EP08 in the OpenCV docs). **[VERIFY link]**
- ORB: Rublee et al., *ORB: An efficient alternative to SIFT or SURF*, ICCV 2011. **[VERIFY link]**

### 4.3 Datasets (for practice data and context)

The brief provides no data, so I need public data or my own. None of these is a perfect fit.

| Dataset | What it is | Useful for me? |
|---|---|---|
| **DeepPCB** | 1,500 image pairs of defect-free templates and aligned test images with six trace-level defect types (open, short, mousebite, spur, pin hole, spurious copper); MIT licence | Mostly bare-board **trace** defects, not missing components. Good as an example of the template-vs-test setup. |
| **FICS-PCB** | 8,685 images of 31 PCBs with 75,965 annotated components (ICs, capacitors, resistors, inductors), with variation in illumination, scale and image sensor; CC BY | Real boards with component labels. Possible source of realistic reference images. |
| **PCB-MC** | A recent dataset adding missing-component labels at expected footprint locations, derived from a Roboflow Universe board dataset | Directly about missing components. **[VERIFY availability and licence]** |
| **PKU-Market-PCB** | Bare-board defect dataset (missing hole, mouse bite, open circuit, short, spur, spurious copper) | Bare-board defects, not component presence. Low relevance. (An AI suggestion described it as "component-related"; my reading of the sources says it is about bare-board defects. **[VERIFY by opening the dataset page]**) |
| **VisA** (PCB1 to PCB4 subsets) | Industrial anomaly dataset from the ECCV 2022 SPot-the-Difference paper: 10,821 images (9,621 normal, 1,200 anomalous) over 12 objects. The four PCB subsets each have about 1,000 normal and 100 anomalous images, labelled "complex structure"; the paper's example PCB defect is molten metal on a board with millimetre-sized components. Dataset licence CC-BY-4.0. | Not a missing-component dataset, since its anomalies are real defects. But the ~1,000 normal photos per board could supply realistic reference images, which I would then edit to remove parts. **[EXPERIMENT: check whether the camera pose is consistent within a PCB subset]** |

Links I found while searching:
- DeepPCB: https://labelbox.com/datasets/deeppcb-printed-circuit-board-pcb-defect-dataset/
- FICS-PCB paper: https://eprint.iacr.org/2020/366 (data: https://www.trust-hub.org/data)
- PCB-MC: https://arxiv.org/pdf/2609.39427 **[VERIFY]**
- PKU-Market-PCB (described in): https://arxiv.org/pdf/2507.17176
- VisA / SPot-the-Difference: https://github.com/amazon-science/spot-diff (paper: https://arxiv.org/abs/2207.14315)

**Plan for test data:** take or generate one good reference image, then create "missing part" variants by editing the *same* image so the camera view is identical. Create "complete" variants with small shifts, brightness changes and noise. Real datasets are used for ideas and sanity checks, not as the main test set. If I generate images with an AI image tool, I will keep the same camera position and view for every image, as the brief requires.

### 4.4 UI framework: Streamlit vs a small web app (FastAPI + plain HTML/JS)

The UI is part of the score, and the user must **draw a box and type a name for each part**, then see results drawn on the image. That drawing step is the hardest UI piece, so it decides the framework.

| | Streamlit + `streamlit-drawable-canvas` | FastAPI/Flask + one HTML page with a `<canvas>` |
|---|---|---|
| Speed to first working UI | Fast | Slower |
| Runs locally with one command | Yes (`streamlit run app.py`) | Yes (one run command) |
| Box drawing and naming | Rectangle mode returns object data; the box-plus-name flow has to be built around the component | Fully custom, so I can build exactly the flow I want (draw, name inline, edit, delete, zoom) |
| Look and feel | Streamlit's look; custom styling is limited | Full control over layout, colours, "OK" green card, red overlays |
| Dependency risk | The canvas is a third-party component. Its release notes say `background_image` broke on newer Streamlit because it used a private Streamlit API, and the new 0.10.0 needs Streamlit 1.53+ and Python 3.10+. Older reports mention background images failing to show in some setups. | Only my own code and standard browser features |
| Image handling | Background image is resized to the canvas size by default (0.10.0 adds a "contain" option to keep aspect ratio); changing it resets the drawing | I control scaling and coordinates, so boxes map exactly to original pixels |
| Reruns | Streamlit reruns the script on interaction, so state needs care | Normal request/response with explicit state |

Sources: streamlit-drawable-canvas v0.10.0 release notes https://github.com/andfanilo/streamlit-drawable-canvas/releases/tag/v0.10.0 ; README https://github.com/VikParuchuri/streamlit-drawable-canvas ; Streamlit forum thread https://discuss.streamlit.io/t/drawable-canvas/3671?page=6 ; GitHub issues https://www.github.com/andfanilo/streamlit-drawable-canvas/issues .

Other points to check: Streamlit sends usage statistics by default, which would need to be switched off to be sure nothing leaves the laptop, and I should confirm everything works with Wi-Fi off. **[VERIFY]**

**Working choice (provisional):** a FastAPI (or Flask) backend with one HTML/JS page, because exact control of the draw-and-name flow and the overall look matters more than speed of setup. If time runs short, Streamlit is the fallback. **[confirm after trying both for the annotation step]**

### 4.5 Supporting libraries

| Library | Suggested use | My assessment |
|---|---|---|
| Python, NumPy, OpenCV | Core | Keep. Everything in sections 2 and 3 is OpenCV + NumPy. |
| scikit-image | SSIM and image comparison | Optional. Only needed if SSIM turns out to add value (Decision 5). Avoid the extra dependency otherwise. |
| Pillow | Image handling | Probably unnecessary, since OpenCV reads, writes and resizes images. Might only be needed for format edge cases. |
| Plotly | Charts and analytics | Not needed for the task. A results view with overlays and a list of missing parts is enough. Plotly can work offline, but it adds weight. **[VERIFY]** |
| SQLite / JSON | Local storage | JSON files are enough to save a reference's named boxes. SQLite only if I later need history across many inspections. |
| Pytest | Testing | Keep. Useful for the alignment and "complete board must say OK" tests. |

---

## 5. OpenCV documentation used

- Geometric transformations tutorial (translation, rotation, affine, perspective, `warpAffine`, `warpPerspective`, `getAffineTransform`, `getPerspectiveTransform`): https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
- Official video tracking reference (`findTransformECC`, `findTransformECCWithMask`, `computeECC`): https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html
- ECC parameters (motion types, parameter counts, matrix sizes): https://docs.opencv.org/5.0/main_modules/structcv_1_1ECCParameters.html
- `findTransformECC` description (area-based alignment, initialisation, exception on non-convergence), via the Rust bindings of the OpenCV docs: https://docs.rs/opencv/0.26.1/opencv/video/fn.find_transform_ecc_1.html
- `estimateAffinePartial2D` (4 DOF, RANSAC): https://docs.rs/crate/opencv/0.23.0/target-redirect/opencv/calib3d/fn.estimate_affine_partial_2d.html
- calib3d function list (`estimateAffine2D`, `estimateAffinePartial2D`, `findHomography`): https://docs.opencv.org/3.3.1/d9/d0c/group__calib3d.html
- Template matching tutorial (six methods, mask support): https://docs.opencv.org/4.13.0/de/da9/tutorial_template_matching.html
- OpenCV forum thread on rigid transforms and `estimateAffinePartial2D`: https://forum.opencv.org/t/how-to-get-a-3-3-rigid-transformation-matrix-by-ransac-of-opencv-using-c/346
- Blog explaining the partial affine (4 DOF) transform: https://nghiaho.com/?p=2208

**Still to read and cite directly (canonical API pages):** `findTransformECC`, `warpAffine`, `findHomography` and `ORB` reference pages on docs.opencv.org for the exact OpenCV version I install. **[VERIFY]** Some links above point to third-party mirrors of the OpenCV docs; I will replace them with the official pages for my installed version.

---

## 6. Open questions

1. How much does real perspective tilt matter for this camera setup? (decides Euclidean/affine vs homography)
2. Is an ORB-based initial guess worth the extra code, or is identity enough for ECC?
3. Do edge-similarity or SSIM signals add value on top of template matching once thresholds are per-part?
4. What is the right way to calibrate each part's threshold from a single reference photo (augmentation of the reference)?
5. Is the box-and-name annotation flow noticeably easier in Streamlit's canvas component or in a custom HTML canvas? (decides the UI framework)
6. Can a VisA PCB normal image serve as a realistic reference photo for my own edited test set?
