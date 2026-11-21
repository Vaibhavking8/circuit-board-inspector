"""
tools/evaluate.py — evaluation harness for BoardInspect.

Loads a labelled dataset (images + ground_truth.json from tools/synthetic.py),
runs the full pipeline on each image, and writes a Markdown metrics report.

Usage:
    python -m tools.evaluate --dataset data/runtime/synthetic \\
                              --reference data/samples/board.png \\
                              --out docs/eval_report.md

The dataset directory must contain ground_truth.json produced by
tools/synthetic.py plus the PNG images referenced within it.

If --parts is omitted the parts are read directly from ground_truth.json,
which is the recommended way to evaluate a synthetic dataset.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow running as `python -m tools.evaluate` from the project root without
# installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boardinspect.config import DEFAULT_CONFIG, Config
from boardinspect.models import (
    ImageStatus,
    Part,
    PartStatus,
    ReferenceSetup,
)
from boardinspect.pipeline import calibrate_reference, inspect_one_with_ref

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImageMetrics:
    """Evaluation result for a single image in the dataset."""

    image_name: str
    source_name: str
    expected_missing: list[str]   # part IDs the dataset says are absent
    predicted_status: str         # ImageStatus.value
    detected_missing: list[str]   # part IDs the pipeline called Missing
    correct_outcome: bool         # pipeline verdict matches ground truth
    false_alarm: bool             # complete board predicted as MissingParts
    missed_parts: list[str]       # expected missing but not detected
    extra_parts: list[str]        # detected as missing but not expected
    runtime_ms: float
    alignment_quality: float


@dataclass
class VariationBreakdown:
    """Per-variation-type counts for the aggregate report."""

    total: int = 0
    correct: int = 0
    false_alarms: int = 0
    missed_parts: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class AggregateMetrics:
    """Dataset-level aggregated metrics."""

    total: int = 0
    correct_outcomes: int = 0
    false_alarms: int = 0            # complete boards called MissingParts
    n_complete_boards: int = 0       # boards with no removed parts
    total_expected_missing: int = 0  # sum of len(expected_missing) across all images
    total_detected_missing: int = 0  # sum of len(detected_missing) across all images
    total_missed: int = 0            # sum of len(missed_parts) — false negatives
    total_extra: int = 0             # sum of len(extra_parts) — false positives
    could_not_verify: int = 0
    total_runtime_ms: float = 0.0
    # source_name → VariationBreakdown
    by_source: dict[str, VariationBreakdown] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct_outcomes / self.total if self.total else 0.0

    @property
    def recall(self) -> float:
        """Fraction of expected-missing parts that were detected."""
        if self.total_expected_missing == 0:
            return 1.0
        detected = self.total_expected_missing - self.total_missed
        return detected / self.total_expected_missing

    @property
    def false_alarm_rate(self) -> float:
        """Fraction of complete boards that triggered a false alarm."""
        if self.n_complete_boards == 0:
            return 0.0
        return self.false_alarms / self.n_complete_boards

    @property
    def mean_runtime_ms(self) -> float:
        return self.total_runtime_ms / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def _load_ground_truth(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[Part]]:
    """Return (image_records, parts) from ground_truth.json.

    image_records: the "images" list from the JSON.
    parts: the Part objects reconstructed from the "parts" metadata.
    """
    gt_path = dataset_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground_truth.json not found in {dataset_dir}")

    doc = json.loads(gt_path.read_text(encoding="utf-8"))

    parts: list[Part] = []
    for p in doc.get("parts", []):
        parts.append(Part(
            id=p["id"],
            name=p["name"],
            box=tuple(p["box"]),  # type: ignore[arg-type]
        ))

    return doc["images"], parts


# ---------------------------------------------------------------------------
# Setup construction
# ---------------------------------------------------------------------------

def _build_setup(
    reference_image: np.ndarray,
    parts: list[Part],
    config: Config = DEFAULT_CONFIG,
) -> ReferenceSetup:
    """Construct and calibrate a ReferenceSetup from an image and part list."""
    _, enc = cv2.imencode(".png", reference_image)
    content_hash = hashlib.sha256(enc.tobytes()).hexdigest()
    h, w = reference_image.shape[:2]

    setup = ReferenceSetup(
        content_hash=content_hash,
        original_width=w,
        original_height=h,
        parts=parts,
        config_snapshot={},
    )
    return calibrate_reference(reference_image, setup, config)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset_dir: Path,
    reference_image: np.ndarray,
    parts: list[Part] | None = None,
    sensitivity: float = 0.5,
    config: Config = DEFAULT_CONFIG,
) -> tuple[list[ImageMetrics], AggregateMetrics]:
    """Run the full pipeline on every image in *dataset_dir* and return metrics.

    Args:
        dataset_dir:     Directory produced by tools/synthetic.py.
        reference_image: BGR uint8 reference board image.
        parts:           Annotated Part list.  If None, parts are read from
                         ground_truth.json (recommended for synthetic datasets).
        sensitivity:     Pipeline sensitivity passed to inspect_one_with_ref.
        config:          Config instance.

    Returns:
        (image_metrics, aggregate) — per-image and aggregated results.
    """
    gt_entries, gt_parts = _load_ground_truth(dataset_dir)

    effective_parts = parts if parts is not None else gt_parts
    if not effective_parts:
        raise ValueError("No parts provided and ground_truth.json contains no parts.")

    setup = _build_setup(reference_image, effective_parts, config)

    image_metrics: list[ImageMetrics] = []
    agg = AggregateMetrics()

    for entry in gt_entries:
        # Resolve image path relative to the dataset directory.
        img_path = dataset_dir / entry["image_path"]
        if not img_path.exists():
            logger.warning("Image not found: %s — skipping.", img_path)
            continue

        raw = img_path.read_bytes()
        expected_missing: list[str] = entry.get("removed_parts", [])
        source_name: str = entry.get("source_name", "")
        image_name: str = entry.get("image_name", img_path.stem)

        # --- run the pipeline ------------------------------------------------
        try:
            result = inspect_one_with_ref(
                setup, reference_image, raw, sensitivity, config
            )
        except Exception as exc:
            logger.warning("Pipeline error on %s: %s", img_path.name, exc)
            m = ImageMetrics(
                image_name=image_name,
                source_name=source_name,
                expected_missing=expected_missing,
                predicted_status=ImageStatus.CouldNotVerify.value,
                detected_missing=[],
                correct_outcome=False,
                false_alarm=False,
                missed_parts=list(expected_missing),
                extra_parts=[],
                runtime_ms=0.0,
                alignment_quality=0.0,
            )
            image_metrics.append(m)
            agg.total += 1
            agg.total_expected_missing += len(expected_missing)
            agg.total_missed += len(expected_missing)
            agg.could_not_verify += 1
            if not expected_missing:
                agg.n_complete_boards += 1
            _update_by_source(agg, source_name, correct=False, false_alarm=False,
                              missed=len(expected_missing))
            continue

        detected_missing = [
            pr.part_id
            for pr in result.part_results
            if pr.status == PartStatus.Missing
        ]

        expected_set = set(expected_missing)
        detected_set = set(detected_missing)
        missed = sorted(expected_set - detected_set)
        extra  = sorted(detected_set - expected_set)
        is_complete = len(expected_missing) == 0

        if is_complete:
            # Correct: pipeline sees no missing parts (OK status)
            correct    = result.status == ImageStatus.OK
            false_alarm = result.status == ImageStatus.MissingParts
        else:
            # Correct: pipeline found all and only the missing parts
            correct     = (result.status == ImageStatus.MissingParts and not missed)
            false_alarm = False

        m = ImageMetrics(
            image_name=image_name,
            source_name=source_name,
            expected_missing=expected_missing,
            predicted_status=result.status.value,
            detected_missing=detected_missing,
            correct_outcome=correct,
            false_alarm=false_alarm,
            missed_parts=missed,
            extra_parts=extra,
            runtime_ms=result.runtime_ms,
            alignment_quality=result.alignment.quality,
        )
        image_metrics.append(m)

        agg.total                  += 1
        agg.correct_outcomes       += int(correct)
        agg.false_alarms           += int(false_alarm)
        agg.total_expected_missing += len(expected_missing)
        agg.total_detected_missing += len(detected_missing)
        agg.total_missed           += len(missed)
        agg.total_extra            += len(extra)
        agg.total_runtime_ms       += result.runtime_ms
        if is_complete:
            agg.n_complete_boards += 1
        if result.status == ImageStatus.CouldNotVerify:
            agg.could_not_verify += 1

        _update_by_source(agg, source_name, correct=correct,
                          false_alarm=false_alarm, missed=len(missed))

    return image_metrics, agg


def _update_by_source(
    agg: AggregateMetrics,
    source_name: str,
    *,
    correct: bool,
    false_alarm: bool,
    missed: int,
) -> None:
    """Update the per-source breakdown in *agg* in-place."""
    bd = agg.by_source.setdefault(source_name, VariationBreakdown())
    bd.total += 1
    bd.correct += int(correct)
    bd.false_alarms += int(false_alarm)
    bd.missed_parts += missed


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(
    image_metrics: list[ImageMetrics],
    agg: AggregateMetrics,
    out_path: Path,
) -> None:
    """Write a Markdown evaluation report to *out_path*.

    The report contains:
    - A summary table of aggregate metrics.
    - A breakdown by source/variation type.
    - A per-image results table.
    """
    lines: list[str] = []

    lines += [
        "# BoardInspect Evaluation Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total images evaluated | {agg.total} |",
        f"| Accuracy (correct outcomes) | {agg.accuracy:.1%} |",
        f"| Part recall | {agg.recall:.1%} |",
        f"| False-alarm rate (complete boards) | {agg.false_alarm_rate:.1%} |",
        f"| Complete boards | {agg.n_complete_boards} |",
        f"| Could not verify | {agg.could_not_verify} |",
        f"| Total expected missing parts | {agg.total_expected_missing} |",
        f"| Missed parts (false negatives) | {agg.total_missed} |",
        f"| Extra flagged (false positives per-part) | {agg.total_extra} |",
        f"| Mean runtime per image | {agg.mean_runtime_ms:.0f} ms |",
        "",
    ]

    if agg.by_source:
        lines += [
            "## Breakdown by source / variation type",
            "",
            "| Source | Images | Accuracy | False alarms | Missed parts |",
            "|---|---|---|---|---|",
        ]
        for src, bd in sorted(agg.by_source.items()):
            lines.append(
                f"| {src or '(unknown)'} | {bd.total} | {bd.accuracy:.1%} "
                f"| {bd.false_alarms} | {bd.missed_parts} |"
            )
        lines.append("")

    lines += [
        "## Per-image results",
        "",
        "| Image | Source | Expected missing | Detected missing | "
        "Status | Correct | Missed | Extra | Runtime ms | Align quality |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in image_metrics:
        exp    = ", ".join(m.expected_missing) or "—"
        det    = ", ".join(m.detected_missing) or "—"
        missed = ", ".join(m.missed_parts)     or "—"
        extra  = ", ".join(m.extra_parts)      or "—"
        ok     = "✓" if m.correct_outcome else "✗"
        lines.append(
            f"| {m.image_name} | {m.source_name} | {exp} | {det} "
            f"| {m.predicted_status} | {ok} | {missed} | {extra} "
            f"| {m.runtime_ms:.0f} | {m.alignment_quality:.3f} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Report written to %s", out_path)
    print(f"Report written to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate BoardInspect on a labelled synthetic dataset."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset directory produced by tools/synthetic.py.",
    )
    parser.add_argument(
        "--reference", required=True,
        help="Path to the reference board image (must match the dataset).",
    )
    parser.add_argument(
        "--out", default="docs/eval_report.md",
        help="Output Markdown report path (default: docs/eval_report.md).",
    )
    parser.add_argument(
        "--sensitivity", type=float, default=0.5,
        help="Pipeline sensitivity in [0, 1] (default: 0.5).",
    )
    args = parser.parse_args(argv)

    ref = cv2.imread(args.reference, cv2.IMREAD_COLOR)
    if ref is None:
        print(f"ERROR: cannot read reference image: {args.reference}",
              file=sys.stderr)
        sys.exit(1)

    # Parts are read from ground_truth.json inside the dataset directory.
    # Pass parts=None so evaluate_dataset falls back to that source.
    metrics, agg = evaluate_dataset(
        Path(args.dataset),
        ref,
        parts=None,
        sensitivity=args.sensitivity,
    )

    write_report(metrics, agg, Path(args.out))
    print(
        f"Accuracy: {agg.accuracy:.1%}  "
        f"Recall: {agg.recall:.1%}  "
        f"False-alarm rate: {agg.false_alarm_rate:.1%}"
    )


if __name__ == "__main__":
    _main()
