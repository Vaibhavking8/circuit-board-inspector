"""
tools/alignment_sweep.py — registration strategy comparison sweep.

Generates synthetic warps with known ground-truth transforms and measures
each strategy's residual alignment error. Results are written to
docs/alignment_sweep.md. Use this to settle [EXPERIMENT] claims in design §3.2.

Usage:
    python -m tools.alignment_sweep --reference data/samples/board.png \\
                                    --out docs/alignment_sweep.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boardinspect.config import DEFAULT_CONFIG
from boardinspect.imaging.preprocess import preprocess
from boardinspect.registration.cascade import REGISTRATION_CASCADE
from boardinspect.registration.ecc import ECC_AFFINE, ECC_EUCLIDEAN
from boardinspect.registration.interface import RegistrationStatus
from boardinspect.registration.orb import ORB_REGISTRAR

logger = logging.getLogger(__name__)

# ── Registrar registry ────────────────────────────────────────────────────────
REGISTRARS = {
    "ecc_euclidean":  ECC_EUCLIDEAN,
    "ecc_affine":     ECC_AFFINE,
    "orb_homography": ORB_REGISTRAR,
    "cascade":        REGISTRATION_CASCADE,
}

# ── Variation types ───────────────────────────────────────────────────────────
# (label, list of (shift_px, rotation_deg, scale))
VARIATIONS = [
    ("shift_5px",     [(5,  0,   1.0), (10, 0,   1.0), (20, 0,   1.0)]),
    ("rotation_3deg", [(0,  3,   1.0), (0,  5,   1.0), (0,  10,  1.0)]),
    ("scale_2pct",    [(0,  0,   1.02),(0,  0,   1.05),(0,  0,   1.10)]),
    ("combined",      [(5,  3,   1.02),(10, 5,   1.05),(15, 8,   1.0)]),
]


@dataclass
class WarpResult:
    strategy: str
    variation: str
    shift_px: float
    rotation_deg: float
    scale: float
    status: str
    residual_px: float    # mean corner displacement; NaN if failed


@dataclass
class SweepResults:
    rows: list[WarpResult] = field(default_factory=list)


# ── Ground-truth residual measurement ────────────────────────────────────────

def _corners(h: int, w: int) -> np.ndarray:
    """Four corners of an HxW image as a (4, 2) float32 array."""
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def _apply_affine(corners: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 2×3 affine matrix to (N, 2) corners; return (N, 2) float32."""
    pts = np.hstack([corners, np.ones((len(corners), 1), dtype=np.float32)])
    return (m @ pts.T).T


def _residual(predicted_transform: np.ndarray, gt_transform: np.ndarray, h: int, w: int) -> float:
    """Mean corner displacement (px) between predicted and ground-truth transforms.

    Both transforms may be 2×3 affine or 3×3 homography.  Homographies are
    handled via cv2.perspectiveTransform; affines via matrix multiplication.
    """
    corners = _corners(h, w)

    # Ground truth is always 2×3 from _make_warp
    gt_pts = _apply_affine(corners, gt_transform)

    # Predicted transform may be 2×3 affine or 3×3 homography
    if predicted_transform.shape == (3, 3):
        # perspectiveTransform expects (1, N, 2)
        pred_pts = cv2.perspectiveTransform(corners.reshape(1, -1, 2), predicted_transform)[0]
    else:
        pred_pts = _apply_affine(corners, predicted_transform)

    return float(np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1)))


def _make_warp(ref_gray: np.ndarray, shift_px: float, rotation_deg: float, scale: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (warped_image, gt_transform_2x3).

    Applies rotate-and-scale about the image centre, then shifts diagonally
    by *shift_px* pixels in both x and y.
    """
    h, w = ref_gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale)
    M[0, 2] += shift_px
    M[1, 2] += shift_px
    warped = cv2.warpAffine(
        ref_gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped, M


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(reference_image: np.ndarray, config=DEFAULT_CONFIG) -> SweepResults:
    """Run every registered strategy against every warp variation.

    Args:
        reference_image: BGR (or grayscale) board image.
        config:          Config instance; all strategy parameters are read from it.

    Returns:
        SweepResults with one WarpResult row per (strategy, variation, params) triple.
    """
    results = SweepResults()
    ref_gray = preprocess(reference_image, config, stage="registration")
    h, w = ref_gray.shape[:2]

    for var_label, params in VARIATIONS:
        for (shift, rot, scale) in params:
            warped, gt = _make_warp(ref_gray, shift, rot, scale)

            for strat_name, registrar in REGISTRARS.items():
                try:
                    result = registrar.register(ref_gray, warped, config)
                    if result.status == RegistrationStatus.Failed:
                        residual = float("nan")
                        status = "failed"
                    else:
                        residual = _residual(result.transform, gt, h, w)
                        status = result.status.value
                except Exception as exc:
                    logger.debug(
                        "%s on shift=%s rot=%s scale=%s: %s",
                        strat_name, shift, rot, scale, exc,
                    )
                    residual = float("nan")
                    status = "error"

                results.rows.append(WarpResult(
                    strategy=strat_name,
                    variation=var_label,
                    shift_px=shift,
                    rotation_deg=rot,
                    scale=scale,
                    status=status,
                    residual_px=residual,
                ))

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(results: SweepResults, out_path: Path) -> None:
    """Write a Markdown report of sweep results to *out_path*.

    The report contains:
    - A per-strategy summary table (mean residual, success rate).
    - A full per-variation breakdown table.

    Args:
        results:  SweepResults from run_sweep().
        out_path: Destination .md file; parent directories are created if needed.
    """
    strategies = list(REGISTRARS.keys())

    lines = [
        "# Alignment Sweep Report\n",
        "Residual = mean corner displacement (px) between predicted and "
        "ground-truth transform.  ",
        "NaN / — = strategy failed or errored on this variation.\n",
        "## Per-strategy summary (mean residual across all variations)\n",
        "| Strategy | Mean residual (px) | Success rate |",
        "|---|---|---|",
    ]

    for strat in strategies:
        strat_rows = [r for r in results.rows if r.strategy == strat]
        valid = [r.residual_px for r in strat_rows if not np.isnan(r.residual_px)]
        mean_res = sum(valid) / len(valid) if valid else float("nan")
        success = len(valid) / len(strat_rows) if strat_rows else 0.0
        mean_str = f"{mean_res:.2f}" if not np.isnan(mean_res) else "—"
        lines.append(f"| {strat} | {mean_str} | {success:.0%} |")

    lines += [
        "",
        "## Per-variation breakdown\n",
        "| Strategy | Variation | Shift (px) | Rotation (°) | Scale | Status | Residual (px) |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results.rows:
        res_str = f"{r.residual_px:.2f}" if not np.isnan(r.residual_px) else "—"
        lines.append(
            f"| {r.strategy} | {r.variation} | {r.shift_px} | {r.rotation_deg} "
            f"| {r.scale} | {r.status} | {res_str} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Alignment sweep report written to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Sweep registration strategies.")
    parser.add_argument("--reference", required=True,
                        help="Path to a reference board image (any format OpenCV reads).")
    parser.add_argument("--out", default="docs/alignment_sweep.md",
                        help="Output Markdown report path.")
    args = parser.parse_args()

    ref = cv2.imread(args.reference, cv2.IMREAD_COLOR)
    if ref is None:
        print(f"ERROR: cannot read {args.reference}", file=sys.stderr)
        sys.exit(1)

    results = run_sweep(ref)
    write_report(results, Path(args.out))


if __name__ == "__main__":
    _main()
