"""
tests/test_golden.py — golden regression tests for the BoardInspect pipeline.

Golden files live in tests/golden/. Each is a JSON file with:
  - "case":             str  — human-readable case name
  - "expected_status":  str  — "ok" | "missing_parts" | "could_not_verify"
  - "expected_missing": list[str]  — part ids whose status == "missing"

If no golden files exist the test module generates them from a small
deterministic synthetic board, runs the pipeline, and writes the golden
files.  Subsequent runs assert the output matches exactly.

To intentionally update goldens (after a deliberate algorithm change):
  1. Delete every .json file in tests/golden/.
  2. Re-run pytest — new goldens are written automatically.
  3. Review the diff and commit them alongside the algorithm change.

Run with:
  pytest tests/test_golden.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make sure the installed package (or src layout) is importable from the
# project root when running without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boardinspect.config import DEFAULT_CONFIG, Config, ImageLimitsConfig
from boardinspect.models import Part, ReferenceSetup
from boardinspect.pipeline import calibrate_reference, inspect_one_with_ref

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "golden"

# Use a looser image-limits config so our small synthetic test images
# (which are intentionally tiny for speed) are accepted by the pipeline.
# Real uploads still hit the full limits enforced at the API layer.
_TEST_CONFIG = Config(
    image_limits=ImageLimitsConfig(
        max_file_size_mb=25.0,
        max_megapixels=40.0,
        recommended_megapixels=12.0,
        min_shorter_side=64,   # allow our small synthetic boards
    )
)

# ---------------------------------------------------------------------------
# Synthetic board factory
# ---------------------------------------------------------------------------

# Board dimensions: 400 wide × 300 tall so the shorter side (300) comfortably
# exceeds the default 240 px minimum and the aspect ratio is easy to reason about.
_W, _H = 400, 300


def _make_synthetic_board(seed: int = 42) -> np.ndarray:
    """Create a deterministic synthetic PCB-like image (300 × 400 BGR).

    Background: mid-tone greenish substrate with mild per-pixel noise so the
    board has non-zero variance (required for TM_CCOEFF_NORMED to behave
    normally).

    Components have *internal* texture — alternating bright/dark lines painted
    over the base region so that each part crop has high internal variance
    (std ≳ 15).  This ensures TM_CCOEFF_NORMED returns a high score (~1.0)
    when the part is present and a clearly low score (≲ 0.1) when it has been
    inpainted away with background texture.  Uniform colour blocks cause
    TM_CCOEFF_NORMED to return degenerate values and cannot be used here.

    Part boxes (normalised x, y, w, h):
      C1: (0.10, 0.10, 0.25, 0.25) → pixels x=40, y=30, w=100, h=75
      C2: (0.60, 0.60, 0.25, 0.25) → pixels x=240, y=180, w=100, h=75
    """
    rng = np.random.default_rng(seed)

    # Textured background: mid-tone green with ±15 noise
    board = rng.integers(60, 100, (_H, _W, 3), dtype=np.uint8)
    board[:, :, 1] = np.clip(
        board[:, :, 1].astype(np.int32) + 30, 0, 255
    ).astype(np.uint8)

    def _textured_part(
        base_lo: int, base_hi: int, h: int, w: int, stripe_val: int
    ) -> np.ndarray:
        """Fill a region with base noise and add horizontal+vertical stripes."""
        patch = rng.integers(base_lo, base_hi, (h, w, 3), dtype=np.uint8)
        # Horizontal stripes every 8 px (clear internal gradient)
        patch[4::8, :, :] = np.clip(
            patch[4::8, :, :].astype(np.int32) + stripe_val, 0, 255
        ).astype(np.uint8)
        # Vertical stripes every 10 px
        patch[:, 5::10, :] = np.clip(
            patch[:, 5::10, :].astype(np.int32) + stripe_val, 0, 255
        ).astype(np.uint8)
        return patch

    # C1: bright component (base 185–220, stripes +35) — top-left quadrant
    board[30:105, 40:140] = _textured_part(185, 220, 75, 100, 35)

    # C2: dark component (base 15–45, stripes −10 clamped)  — bottom-right
    # For C2 stripes we subtract rather than add so internal contrast remains.
    c2_patch = rng.integers(20, 50, (75, 100, 3), dtype=np.uint8)
    c2_patch[4::8, :, :] = np.clip(
        c2_patch[4::8, :, :].astype(np.int32) + 30, 0, 255
    ).astype(np.uint8)
    c2_patch[:, 5::10, :] = np.clip(
        c2_patch[:, 5::10, :].astype(np.int32) + 30, 0, 255
    ).astype(np.uint8)
    board[180:255, 240:340] = c2_patch

    return board


# Normalised part boxes tuned to the board above.
# box = (x, y, w, h) in [0, 1] relative to reference image dimensions.
_PARTS: list[Part] = [
    Part(id="c1", name="C1", box=(0.10, 0.10, 0.25, 0.25)),
    Part(id="c2", name="C2", box=(0.60, 0.60, 0.25, 0.25)),
]

_PARTS_BY_ID: dict[str, Part] = {p.id: p for p in _PARTS}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_png(image: np.ndarray) -> bytes:
    """Encode *image* (BGR uint8) to PNG bytes."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _make_missing(ref: np.ndarray, part: Part) -> np.ndarray:
    """Return a copy of *ref* with *part*'s region inpainted (component removed).

    Uses cv2.INPAINT_TELEA which fills the masked region from surrounding
    pixels, reproducing a bare-board appearance convincingly.
    """
    h, w = ref.shape[:2]
    bx = round(part.box[0] * w)
    by = round(part.box[1] * h)
    bw = round(part.box[2] * w)
    bh = round(part.box[3] * h)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[by : by + bh, bx : bx + bw] = 255
    return cv2.inpaint(ref, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def _build_setup(ref: np.ndarray) -> ReferenceSetup:
    """Build and calibrate a ReferenceSetup from *ref* with the two test parts."""
    _, enc = cv2.imencode(".png", ref)
    content_hash = hashlib.sha256(enc.tobytes()).hexdigest()
    setup = ReferenceSetup(
        content_hash=content_hash,
        original_width=ref.shape[1],
        original_height=ref.shape[0],
        parts=_PARTS,
        config_snapshot={},
    )
    return calibrate_reference(ref, setup, _TEST_CONFIG)


# ---------------------------------------------------------------------------
# Golden file management
# ---------------------------------------------------------------------------

def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def _write_golden(name: str, data: dict) -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    _golden_path(name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_golden(name: str) -> dict:
    return json.loads(_golden_path(name).read_text(encoding="utf-8"))


def _generate_goldens(setup: ReferenceSetup, ref: np.ndarray) -> None:
    """Run the pipeline on known inputs and write golden JSON files.

    Called once automatically when no golden files exist.  The results
    define the expected behaviour; running the suite again verifies they
    have not changed.
    """
    sensitivity = 0.5

    # Case 1: complete board — reference image inspected against itself.
    result_ok = inspect_one_with_ref(
        setup, ref, _encode_png(ref), sensitivity, _TEST_CONFIG
    )
    _write_golden("complete_board", {
        "case": "complete_board",
        "expected_status": result_ok.status.value,
        "expected_missing": [],
        "note": "reference inspected against itself; all parts should be present",
    })

    # Case 2: C1 removed.
    missing_c1 = _make_missing(ref, _PARTS_BY_ID["c1"])
    result_m1 = inspect_one_with_ref(
        setup, ref, _encode_png(missing_c1), sensitivity, _TEST_CONFIG
    )
    _write_golden("missing_c1", {
        "case": "missing_c1",
        "expected_status": result_m1.status.value,
        "expected_missing": [
            pr.part_id for pr in result_m1.part_results
            if pr.status.value == "missing"
        ],
        "note": "C1 region inpainted; pipeline should flag c1 as missing",
    })

    # Case 3: C2 removed.
    missing_c2 = _make_missing(ref, _PARTS_BY_ID["c2"])
    result_m2 = inspect_one_with_ref(
        setup, ref, _encode_png(missing_c2), sensitivity, _TEST_CONFIG
    )
    _write_golden("missing_c2", {
        "case": "missing_c2",
        "expected_status": result_m2.status.value,
        "expected_missing": [
            pr.part_id for pr in result_m2.part_results
            if pr.status.value == "missing"
        ],
        "note": "C2 region inpainted; pipeline should flag c2 as missing",
    })

    print(f"\nGenerated golden files in {GOLDEN_DIR}")


# ---------------------------------------------------------------------------
# Module-scoped fixture — shared across all tests in this file.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref_and_setup():
    """Return (ref_image, calibrated_setup) and generate goldens if absent."""
    ref = _make_synthetic_board(seed=42)
    setup = _build_setup(ref)
    if not list(GOLDEN_DIR.glob("*.json")):
        _generate_goldens(setup, ref)
    return ref, setup


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_complete_board(ref_and_setup):
    """Inspecting the reference against itself must reproduce the golden status.

    The reference is a known-good board, so either 'ok' or the status recorded
    in the golden file at generation time must be returned unchanged (R9.4).
    """
    ref, setup = ref_and_setup
    golden = _read_golden("complete_board")
    result = inspect_one_with_ref(
        setup, ref, _encode_png(ref), 0.5, _TEST_CONFIG
    )
    assert result.status.value == golden["expected_status"], (
        f"complete_board: expected '{golden['expected_status']}', "
        f"got '{result.status.value}'"
    )


def test_missing_c1(ref_and_setup):
    """Removing C1 must reproduce the golden status and missing-part set.

    Both the overall image status and the per-part 'missing' set are locked.
    If a future algorithm change causes a different number of false positives
    or false negatives, this test fails and the golden must be deliberately
    updated.
    """
    ref, setup = ref_and_setup
    golden = _read_golden("missing_c1")
    missing = _make_missing(ref, _PARTS_BY_ID["c1"])
    result = inspect_one_with_ref(
        setup, ref, _encode_png(missing), 0.5, _TEST_CONFIG
    )
    assert result.status.value == golden["expected_status"], (
        f"missing_c1: status expected '{golden['expected_status']}', "
        f"got '{result.status.value}'"
    )
    detected = {pr.part_id for pr in result.part_results if pr.status.value == "missing"}
    expected = set(golden["expected_missing"])
    assert detected == expected, (
        f"missing_c1: missing-part set expected {expected}, got {detected}"
    )


def test_missing_c2(ref_and_setup):
    """Removing C2 must reproduce the golden status and missing-part set."""
    ref, setup = ref_and_setup
    golden = _read_golden("missing_c2")
    missing = _make_missing(ref, _PARTS_BY_ID["c2"])
    result = inspect_one_with_ref(
        setup, ref, _encode_png(missing), 0.5, _TEST_CONFIG
    )
    assert result.status.value == golden["expected_status"], (
        f"missing_c2: status expected '{golden['expected_status']}', "
        f"got '{result.status.value}'"
    )
    detected = {pr.part_id for pr in result.part_results if pr.status.value == "missing"}
    expected = set(golden["expected_missing"])
    assert detected == expected, (
        f"missing_c2: missing-part set expected {expected}, got {detected}"
    )


def test_determinism(ref_and_setup):
    """Identical inputs must produce identical results on consecutive runs (R9.4).

    Covers complete-board and missing-C1 cases to exercise both the
    'all present' and 'some missing' code paths.
    """
    ref, setup = ref_and_setup
    encoded = _encode_png(ref)

    r1 = inspect_one_with_ref(setup, ref, encoded, 0.5, _TEST_CONFIG)
    r2 = inspect_one_with_ref(setup, ref, encoded, 0.5, _TEST_CONFIG)
    assert r1.status == r2.status, "Complete board: status not deterministic"
    assert all(
        a.status == b.status and a.part_id == b.part_id
        for a, b in zip(r1.part_results, r2.part_results)
    ), "Complete board: per-part results not deterministic"

    missing = _make_missing(ref, _PARTS_BY_ID["c1"])
    encoded_m = _encode_png(missing)
    r3 = inspect_one_with_ref(setup, ref, encoded_m, 0.5, _TEST_CONFIG)
    r4 = inspect_one_with_ref(setup, ref, encoded_m, 0.5, _TEST_CONFIG)
    assert r3.status == r4.status, "Missing-C1 board: status not deterministic"
    assert all(
        a.status == b.status and a.part_id == b.part_id
        for a, b in zip(r3.part_results, r4.part_results)
    ), "Missing-C1 board: per-part results not deterministic"


def test_wrong_aspect_ratio_gives_could_not_verify(ref_and_setup):
    """An image with a very different aspect ratio must yield CouldNotVerify.

    The reference is 400×300 (AR ≈ 1.33).  Supplying a 300×600 portrait image
    (AR = 0.50) gives a relative difference of about 62 %, well above the 15 %
    tolerance, so the pipeline must reject it without calling alignment or
    returning OK (R9.1, R9.2).
    """
    ref, setup = ref_and_setup
    # 300 wide × 600 tall — portrait orientation, very different AR from 400×300
    wrong = np.zeros((600, 300, 3), dtype=np.uint8)
    result = inspect_one_with_ref(
        setup, ref, _encode_png(wrong), 0.5, _TEST_CONFIG
    )
    assert result.status.value == "could_not_verify", (
        f"Wrong-AR image: expected 'could_not_verify', got '{result.status.value}'"
    )
