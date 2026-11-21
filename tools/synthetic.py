"""
tools/synthetic.py — deterministic synthetic dataset generator for BoardInspect.

Generates augmented board images (complete and with selected parts removed)
for calibration, evaluation, and demo purposes.

Usage:
    python tools/synthetic.py --reference path/to/ref.png --out data/runtime/synthetic
    python tools/synthetic.py --reference path/to/ref.png --out data/runtime/synthetic --seed 7
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing boardinspect from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import dataclasses
import itertools
import json
import logging
from typing import Sequence

import cv2
import numpy as np

from boardinspect.config import DEFAULT_CONFIG, Config
from boardinspect.coordinates import normalized_to_pixel
from boardinspect.models import Part

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VariationInfo:
    """Records all augmentation parameters applied to one image."""
    shift_px: tuple[int, int]
    rotation_deg: float
    brightness: float
    contrast: float
    noise_std: float
    blur_applied: bool
    jpeg_quality: int | None  # None = PNG (lossless)


@dataclasses.dataclass
class GroundTruth:
    """Ground-truth record for one generated image in the dataset."""
    image_path: str                   # relative path inside output_dir; filled by generate_dataset
    source_name: str                  # "complete", "missing_C1", etc.
    removed_parts: list[str]          # part ids that were inpainted
    removed_boxes_px: list[list[int]] # (x, y, w, h) in pixels for each removed part
    variation: VariationInfo


# ---------------------------------------------------------------------------
# Internal augmentation helpers
# ---------------------------------------------------------------------------

def _augment(
    image: np.ndarray,
    rng: np.random.Generator,
    config: Config,
) -> tuple[np.ndarray, VariationInfo]:
    """Apply randomised photometric and geometric augmentation.

    All parameter ranges are taken from ``config.calibration``.  The
    function is pure: it does not modify *image* in-place.

    Args:
        image:  BGR uint8 input (H × W × 3) or greyscale (H × W).
        rng:    Seeded NumPy Generator — caller owns the seed.
        config: Configuration object whose ``calibration`` sub-config
                controls the augmentation ranges.

    Returns:
        ``(augmented_image, VariationInfo)`` with the same dtype and shape.
    """
    cal = config.calibration
    h, w = image.shape[:2]
    img = image.copy().astype(np.float32)

    # --- photometric ---------------------------------------------------------
    # Brightness: additive shift sampled from U(-b, +b) * 255
    b_shift = float(rng.uniform(-cal.brightness, cal.brightness) * 255.0)
    # Contrast: multiplicative scale sampled from U(1-c, 1+c)
    c_scale = float(rng.uniform(1.0 - cal.contrast, 1.0 + cal.contrast))

    img = img * c_scale + b_shift
    img = np.clip(img, 0.0, 255.0)

    # --- Gaussian noise -------------------------------------------------------
    noise = rng.normal(0.0, cal.noise_std, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0.0, 255.0)

    # --- geometric ------------------------------------------------------------
    shift_x = int(rng.integers(-cal.shift_px, cal.shift_px + 1))
    shift_y = int(rng.integers(-cal.shift_px, cal.shift_px + 1))
    rot_deg = float(rng.uniform(-cal.rotation_degrees, cal.rotation_degrees))

    cx, cy = w / 2.0, h / 2.0
    # Combined rotation + translation matrix
    M = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    M[0, 2] += shift_x
    M[1, 2] += shift_y
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # --- optional blur --------------------------------------------------------
    # Apply blur with 50 % probability (driven by the same rng for determinism).
    apply_blur = bool(rng.integers(0, 2))
    if apply_blur:
        k = cal.blur_kernel
        # Kernel size must be positive and odd.
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    result = np.clip(img, 0.0, 255.0).astype(np.uint8)

    info = VariationInfo(
        shift_px=(shift_x, shift_y),
        rotation_deg=rot_deg,
        brightness=b_shift / 255.0,
        contrast=c_scale,
        noise_std=cal.noise_std,
        blur_applied=apply_blur,
        jpeg_quality=None,
    )
    return result, info


def _inpaint_parts(
    image: np.ndarray,
    parts: list[Part],
    part_indices: Sequence[int],
    image_w: int,
    image_h: int,
    config: Config,
) -> np.ndarray:
    """Return a copy of *image* with the selected parts removed via inpainting.

    Uses ``cv2.inpaint`` with the TELEA algorithm (radius = 5 px).  The
    inpaint mask is built from the pixel boxes of the requested parts.

    Args:
        image:        BGR uint8 source image (H × W × 3).
        parts:        Full list of annotated parts.
        part_indices: Indices into *parts* that should be removed.
        image_w:      Width of *image* in pixels.
        image_h:      Height of *image* in pixels.
        config:       Configuration (not currently used here but kept for
                      future threshold overrides).

    Returns:
        BGR uint8 image with the selected regions inpainted.
    """
    if not part_indices:
        return image.copy()

    mask = np.zeros((image_h, image_w), dtype=np.uint8)

    for idx in part_indices:
        part = parts[idx]
        px, py, pw, ph = normalized_to_pixel(part.box, image_w, image_h)
        # Clamp to valid bounds (should already be valid but be defensive).
        px = max(0, min(px, image_w - 1))
        py = max(0, min(py, image_h - 1))
        pw = max(1, min(pw, image_w - px))
        ph = max(1, min(ph, image_h - py))
        mask[py : py + ph, px : px + pw] = 255

    inpainted = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_variation(
    reference: np.ndarray,
    config: Config = DEFAULT_CONFIG,
    seed: int = 42,
) -> tuple[np.ndarray, VariationInfo]:
    """Generate one augmented copy of *reference*.

    Deterministic: same *seed* always produces the same output.

    Args:
        reference:  BGR uint8 reference image.
        config:     Configuration with calibration parameters.
        seed:       Integer seed for the random generator.

    Returns:
        ``(augmented_image, VariationInfo)``.
    """
    rng = np.random.default_rng(seed)
    return _augment(reference, rng, config)


def generate_missing(
    reference: np.ndarray,
    parts: list[Part],
    part_indices_to_remove: Sequence[int],
    config: Config = DEFAULT_CONFIG,
    seed: int = 42,
) -> tuple[np.ndarray, GroundTruth]:
    """Generate a board image with selected parts removed and augmented.

    The pipeline is: inpaint → augment.

    Args:
        reference:              BGR uint8 reference image.
        parts:                  Full list of annotated parts.
        part_indices_to_remove: Indices into *parts* to remove.
        config:                 Configuration.
        seed:                   Integer seed for determinism.

    Returns:
        ``(result_image, GroundTruth)`` where ``GroundTruth.image_path`` is
        ``""`` — it is filled in by :func:`generate_dataset`.
    """
    h, w = reference.shape[:2]
    rng = np.random.default_rng(seed)

    inpainted = _inpaint_parts(reference, parts, part_indices_to_remove, w, h, config)
    augmented, variation = _augment(inpainted, rng, config)

    removed_part_ids = [parts[i].id for i in part_indices_to_remove]
    removed_boxes: list[list[int]] = []
    for i in part_indices_to_remove:
        px, py, pw, ph = normalized_to_pixel(parts[i].box, w, h)
        removed_boxes.append([px, py, pw, ph])

    # Build a human-readable source name.
    if removed_part_ids:
        source_name = "missing_" + "_".join(parts[i].name for i in part_indices_to_remove)
    else:
        source_name = "complete"

    gt = GroundTruth(
        image_path="",
        source_name=source_name,
        removed_parts=removed_part_ids,
        removed_boxes_px=removed_boxes,
        variation=variation,
    )
    return augmented, gt


def generate_dataset(
    reference: np.ndarray,
    parts: list[Part],
    config: Config = DEFAULT_CONFIG,
    output_dir: str = "data/runtime/synthetic",
    seed: int = 42,
) -> None:
    """Generate and write a full synthetic dataset to *output_dir*.

    Written files::

        complete_000.png
        missing_1_{PartName}_{idx:03d}.png   — one missing-part image per part
        missing_2_{P1}_{P2}_{idx:03d}.png    — up to 3 two-part-missing pairs
        demo_complete.png
        demo_missing_1.png                   — first part missing
        demo_missing_2.png                   — first two parts missing
        ground_truth.json

    ``ground_truth.json`` has the structure::

        {
            "seed": <seed>,
            "parts": [{"id": ..., "name": ..., "box": [...]}],
            "images": [<GroundTruth as dict>, ...]
        }

    All images are written as PNG (lossless).

    Args:
        reference:  BGR uint8 reference image.
        parts:      Annotated parts.
        config:     Configuration.
        output_dir: Directory to write into (created if absent).
        seed:       Master seed — split deterministically per image.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # Use a master RNG to derive per-image seeds so adding/removing images
    # never changes seeds for unrelated images.
    master_rng = np.random.default_rng(seed)

    def _next_seed() -> int:
        return int(master_rng.integers(0, 2**31))

    def _write(img: np.ndarray, filename: str) -> Path:
        dest = out / filename
        cv2.imwrite(str(dest), img)
        return dest

    def _gt_to_dict(gt: GroundTruth) -> dict:
        return {
            "image_path": gt.image_path,
            "source_name": gt.source_name,
            "removed_parts": gt.removed_parts,
            "removed_boxes_px": gt.removed_boxes_px,
            "variation": dataclasses.asdict(gt.variation),
        }

    # --- complete_000.png ----------------------------------------------------
    img_complete, vi_complete = generate_variation(reference, config, seed=_next_seed())
    _write(img_complete, "complete_000.png")
    records.append(_gt_to_dict(GroundTruth(
        image_path="complete_000.png",
        source_name="complete",
        removed_parts=[],
        removed_boxes_px=[],
        variation=vi_complete,
    )))
    logger.debug("Wrote complete_000.png")

    # --- one-part-missing images ---------------------------------------------
    for i, part in enumerate(parts):
        filename = f"missing_1_{part.name}_{i:03d}.png"
        img_m, gt = generate_missing(reference, parts, [i], config, seed=_next_seed())
        gt.image_path = filename
        _write(img_m, filename)
        records.append(_gt_to_dict(gt))
        logger.debug("Wrote %s", filename)

    # --- two-part-missing images (up to 3 pairs) ----------------------------
    pair_count = 0
    for i, j in itertools.combinations(range(len(parts)), 2):
        if pair_count >= 3:
            break
        p1_name = parts[i].name
        p2_name = parts[j].name
        filename = f"missing_2_{p1_name}_{p2_name}_{pair_count:03d}.png"
        img_m2, gt2 = generate_missing(reference, parts, [i, j], config, seed=_next_seed())
        gt2.image_path = filename
        _write(img_m2, filename)
        records.append(_gt_to_dict(gt2))
        logger.debug("Wrote %s", filename)
        pair_count += 1

    # --- demo images ---------------------------------------------------------
    # demo_complete.png
    img_demo_c, vi_demo_c = generate_variation(reference, config, seed=_next_seed())
    _write(img_demo_c, "demo_complete.png")
    records.append(_gt_to_dict(GroundTruth(
        image_path="demo_complete.png",
        source_name="complete",
        removed_parts=[],
        removed_boxes_px=[],
        variation=vi_demo_c,
    )))
    logger.debug("Wrote demo_complete.png")

    # demo_missing_1.png — first part missing
    if parts:
        img_dm1, gt_dm1 = generate_missing(reference, parts, [0], config, seed=_next_seed())
        gt_dm1.image_path = "demo_missing_1.png"
        _write(img_dm1, "demo_missing_1.png")
        records.append(_gt_to_dict(gt_dm1))
        logger.debug("Wrote demo_missing_1.png")

    # demo_missing_2.png — first two parts missing (or just first if only one part)
    if parts:
        indices = [0, 1] if len(parts) >= 2 else [0]
        img_dm2, gt_dm2 = generate_missing(reference, parts, indices, config, seed=_next_seed())
        gt_dm2.image_path = "demo_missing_2.png"
        _write(img_dm2, "demo_missing_2.png")
        records.append(_gt_to_dict(gt_dm2))
        logger.debug("Wrote demo_missing_2.png")

    # --- ground_truth.json ---------------------------------------------------
    parts_meta = [{"id": p.id, "name": p.name, "box": list(p.box)} for p in parts]
    gt_doc = {
        "seed": seed,
        "parts": parts_meta,
        "images": records,
    }
    gt_path = out / "ground_truth.json"
    gt_path.write_text(json.dumps(gt_doc, indent=2), encoding="utf-8")
    logger.info("Dataset written to %s (%d images)", out, len(records))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_demo_parts(image: np.ndarray) -> list[Part]:
    """Create four demo parts covering the four quadrants of *image*."""
    # Each part covers roughly a quarter of the image, slightly inset so the
    # inpaint radius does not bleed outside the image boundary.
    inset = 0.02  # 2 % inset from each edge
    half = 0.5

    return [
        Part(id="tl", name="TL", box=(inset, inset, half - 2 * inset, half - 2 * inset)),
        Part(id="tr", name="TR", box=(half + inset, inset, half - 2 * inset, half - 2 * inset)),
        Part(id="bl", name="BL", box=(inset, half + inset, half - 2 * inset, half - 2 * inset)),
        Part(id="br", name="BR", box=(half + inset, half + inset, half - 2 * inset, half - 2 * inset)),
    ]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for synthetic dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic BoardInspect dataset from a reference image."
    )
    parser.add_argument(
        "--reference",
        required=True,
        metavar="PATH",
        help="Path to the reference board image (PNG or JPEG).",
    )
    parser.add_argument(
        "--out",
        default="data/runtime/synthetic",
        metavar="DIR",
        help="Output directory for generated images and ground_truth.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for deterministic generation (default: 42).",
    )
    args = parser.parse_args(argv)

    ref_path = Path(args.reference)
    if not ref_path.exists():
        parser.error(f"Reference image not found: {ref_path}")

    reference = cv2.imread(str(ref_path))
    if reference is None:
        parser.error(f"Could not decode image: {ref_path}")

    parts = _build_demo_parts(reference)
    generate_dataset(reference, parts, DEFAULT_CONFIG, args.out, seed=args.seed)
    print(f"Dataset written to: {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
