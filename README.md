# BoardInspect

Offline desktop web app that finds **missing components** on printed circuit boards (PCBs) by comparing inspection photos against one reference photo of a good board. No internet, no cloud, no deep learning — just classical computer vision on your laptop.

## Quick start

```bash
# 1. Create a virtual environment and install
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# 2. Run
python run.py
```

The server starts at **http://127.0.0.1:8000** and the browser opens automatically.
If the browser does not open, copy the URL and paste it manually.

## Offline install (no internet on the demo laptop)

Download all wheels on a connected machine:

```bash
pip download -r requirements.txt -d wheels/
```

Copy the `wheels/` folder and `requirements.txt` to the demo laptop, then install:

```bash
pip install --no-index --find-links=wheels/ -r requirements.txt
```

## Verify install

After installing, run the verification script to confirm everything is working:

```bash
python tools/verify_install.py
```

All checks should pass before attempting the demo.

## Usage

1. **Reference** — drag and drop a photo of a complete, good board.
2. **Name Parts** — draw a rectangle around each component and type its name. Click **Calibrate & Continue** when done.
3. **Inspect** — drag and drop 1–3 inspection photos. Each result card shows:
   - ✅ **OK** — no missing parts.
   - 🔴 **Missing Parts** — lists every missing component with a red box on the image.
   - ⚠️ **Could Not Verify** — alignment failed; check the photo orientation.
4. **Sensitivity** — drag the slider (lenient ← → strict) to adjust the detection threshold.
5. **Start over** — resets all state.

## Project structure

| Path | Purpose |
|---|---|
| `run.py` | One-command server launcher |
| `src/boardinspect/` | Python package (CV + API) |
| `src/boardinspect/api/` | FastAPI routes, schemas, jobs |
| `src/boardinspect/imaging/` | Image load, decode, preprocess |
| `src/boardinspect/registration/` | ECC + ORB alignment cascade |
| `src/boardinspect/comparison/` | Template + edge signals |
| `src/boardinspect/decision/` | Decision engine, calibration, sensitivity |
| `src/boardinspect/pipeline.py` | Orchestrates one inspection |
| `src/boardinspect/storage.py` | Save/load reference setups |
| `src/boardinspect/viz.py` | Server-side result overlays |
| `static/` | Frontend (HTML + CSS + JS, no build step) |
| `tools/` | Synthetic data generator, evaluation harness |
| `tests/` | Unit, integration, golden regression tests |
| `data/samples/` | Small demo/test images (committed) |
| `data/runtime/` | Uploads and saved setups (gitignored) |

## Tech stack

- **Python 3.10+**, FastAPI, Uvicorn — backend server (loopback only)
- **OpenCV** (`opencv-python-headless`), NumPy — computer vision
- **Vanilla HTML/CSS/ES modules** — frontend, no framework, no build step
- JSON + PNG on local disk — storage

## Development

```bash
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run specific test files
pytest tests/test_golden.py -v
pytest tests/test_config.py tests/test_errors.py tests/test_models.py -v

# Generate synthetic evaluation dataset
python -m tools.synthetic --reference data/samples/board.png --out data/runtime/synthetic --seed 42

# Run evaluation
python -m tools.evaluate --dataset data/runtime/synthetic --reference data/samples/board.png --out docs/eval_report.md

# Registration strategy sweep
python -m tools.alignment_sweep --reference data/samples/board.png --out docs/alignment_sweep.md
```

## Project docs

- [AI_COLLAB.md](AI_COLLAB.md): AI collaboration log — decisions, reasoning, chat history
- [RESEARCH.md](RESEARCH.md): what was investigated and why each approach was chosen
- [ARCHITECTURE.md](ARCHITECTURE.md): system design, component diagram, and trade-offs

## Licence

[TODO: add licence]
