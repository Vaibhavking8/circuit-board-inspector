"""
storage.py — save and load reference setups (JSON + image) to/from local disk.

Setups are stored under data/runtime/setups/{setup_id}/:
  - setup.json  — JSON with metadata, parts (normalised boxes), calibration, config snapshot
  - reference.png — the reference image (lossless PNG)

The setup_id is derived from the SHA-256 hex digest of the reference image bytes
(first 16 chars), which is stable across saves and can detect image tampering.

No web-framework imports. Raises StorageError for all storage failures.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.errors import StorageError
from boardinspect.models import (
    CalibrationResult,
    Part,
    ReferenceSetup,
    SCHEMA_VERSION,
    SignalStats,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage root — resolved relative to the package root (src/boardinspect/storage.py)
# ---------------------------------------------------------------------------


def _storage_root(config: Config) -> Path:
    """Return the base path for all saved setups."""
    # Resolve relative to the project root (two levels above src/boardinspect/)
    here = Path(__file__).resolve().parent          # src/boardinspect/
    project_root = here.parent.parent               # circuit-board-inspector/
    return project_root / "data" / "runtime" / "setups"


# ---------------------------------------------------------------------------
# SetupSummary
# ---------------------------------------------------------------------------

@dataclass
class SetupSummary:
    """Lightweight summary of a saved reference setup."""
    setup_id: str
    part_count: int
    created_at: float    # Unix timestamp


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _calibration_to_dict(calibration: dict[str, CalibrationResult] | None) -> dict | None:
    """Convert calibration dict to a plain JSON-serialisable structure."""
    if calibration is None:
        return None
    result = {}
    for part_id, cal in calibration.items():
        signal_stats_raw = {}
        for sig_name, conditions in cal.signal_stats.items():
            signal_stats_raw[sig_name] = {
                cond: {"mean": ss.mean, "std": ss.std}
                for cond, ss in conditions.items()
            }
        result[part_id] = {
            "signal_stats": signal_stats_raw,
            "threshold": cal.threshold,
            "reliable": cal.reliable,
            "reason": cal.reason,
        }
    return result


def _calibration_from_dict(raw: dict | None) -> dict[str, CalibrationResult] | None:
    """Reconstruct calibration dict from JSON-loaded data."""
    if raw is None:
        return None
    result = {}
    for part_id, cal_raw in raw.items():
        signal_stats = {
            sig_name: {
                cond: SignalStats(mean=ss["mean"], std=ss["std"])
                for cond, ss in conditions.items()
            }
            for sig_name, conditions in cal_raw["signal_stats"].items()
        }
        result[part_id] = CalibrationResult(
            signal_stats=signal_stats,
            threshold=cal_raw["threshold"],
            reliable=cal_raw["reliable"],
            reason=cal_raw.get("reason", ""),
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_setup(
    reference_setup: ReferenceSetup,
    reference_image_bytes: bytes,
    config: Config = DEFAULT_CONFIG,
) -> str:
    """Save a reference setup and its image to disk.

    Args:
        reference_setup:       Populated ReferenceSetup (with parts, optional calibration).
        reference_image_bytes: Raw bytes of the reference image (any decodable format).
        config:                Config instance (used only for storage path).

    Returns:
        setup_id string (first 16 chars of SHA-256 of the image bytes).

    Raises:
        StorageError: If writing fails.
    """
    setup_id = reference_setup.content_hash[:16]
    storage_dir = _storage_root(config) / setup_id
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(
            f"Could not create storage directory for setup '{setup_id}'.",
            detail=str(exc),
        ) from exc

    # Decode and save image as lossless PNG
    arr = np.frombuffer(reference_image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise StorageError(
            "Could not decode the reference image for saving.",
            detail=f"cv2.imdecode returned None for {len(reference_image_bytes)} bytes",
        )
    image_path = storage_dir / "reference.png"
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise StorageError(
            f"Failed to write reference image to '{image_path}'.",
            detail="cv2.imwrite returned False",
        )

    # Build JSON metadata
    metadata = {
        "setup_id": setup_id,
        "schema_version": reference_setup.schema_version,
        "content_hash": reference_setup.content_hash,
        "original_width": reference_setup.original_width,
        "original_height": reference_setup.original_height,
        "created_at": time.time(),
        "parts": [
            {
                "id": p.id,
                "name": p.name,
                "box": list(p.box),
            }
            for p in reference_setup.parts
        ],
        "calibration": _calibration_to_dict(reference_setup.calibration),
        "config_snapshot": reference_setup.config_snapshot,
    }

    json_path = storage_dir / "setup.json"
    try:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
    except OSError as exc:
        raise StorageError(
            f"Could not write setup metadata to '{json_path}'.",
            detail=str(exc),
        ) from exc

    logger.debug(
        "Saved setup '%s' with %d parts to '%s'.",
        setup_id,
        len(reference_setup.parts),
        storage_dir,
    )
    return setup_id


def load_setup(
    setup_id: str,
    config: Config = DEFAULT_CONFIG,
) -> tuple[ReferenceSetup, np.ndarray]:
    """Load a saved reference setup from disk.

    Args:
        setup_id: The setup identifier returned by save_setup.
        config:   Config instance.

    Returns:
        Tuple (ReferenceSetup, reference_image_array).

    Raises:
        StorageError: If files are missing, corrupt, or have wrong schema version.
    """
    storage_dir = _storage_root(config) / setup_id
    json_path = storage_dir / "setup.json"
    image_path = storage_dir / "reference.png"

    if not json_path.exists():
        raise StorageError(
            f"Setup '{setup_id}' not found.",
            detail=f"Missing file: {json_path}",
        )
    if not image_path.exists():
        raise StorageError(
            f"Reference image for setup '{setup_id}' is missing.",
            detail=f"Missing file: {image_path}",
        )

    # Load JSON
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise StorageError(
            f"Setup '{setup_id}' metadata is corrupt or unreadable.",
            detail=str(exc),
        ) from exc

    # Schema version check
    saved_version = metadata.get("schema_version")
    if saved_version != SCHEMA_VERSION:
        raise StorageError(
            f"Setup '{setup_id}' was saved with schema version {saved_version}, "
            f"but the current version is {SCHEMA_VERSION}. "
            "Please re-annotate the reference.",
            detail=f"Schema mismatch: saved={saved_version}, current={SCHEMA_VERSION}",
        )

    # Reconstruct parts
    try:
        parts = [
            Part(
                id=p["id"],
                name=p["name"],
                box=tuple(p["box"]),
            )
            for p in metadata["parts"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageError(
            f"Setup '{setup_id}' has corrupt part data.",
            detail=str(exc),
        ) from exc

    calibration = _calibration_from_dict(metadata.get("calibration"))

    setup = ReferenceSetup(
        content_hash=metadata["content_hash"],
        original_width=metadata["original_width"],
        original_height=metadata["original_height"],
        parts=parts,
        config_snapshot=metadata.get("config_snapshot", {}),
        schema_version=metadata["schema_version"],
        calibration=calibration,
    )

    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise StorageError(
            f"Could not decode reference image for setup '{setup_id}'.",
            detail=f"cv2.imread returned None for {image_path}",
        )

    logger.debug("Loaded setup '%s' with %d parts.", setup_id, len(parts))
    return setup, image


def list_setups(
    config: Config = DEFAULT_CONFIG,
) -> list[SetupSummary]:
    """List all saved setups.

    Returns an empty list if the storage directory does not exist.
    Silently skips unreadable directories.

    Args:
        config: Config instance.

    Returns:
        List of SetupSummary sorted by creation time (newest first).
    """
    storage_root = _storage_root(config)
    if not storage_root.exists():
        return []

    summaries: list[SetupSummary] = []
    for setup_dir in storage_root.iterdir():
        if not setup_dir.is_dir():
            continue
        json_path = setup_dir / "setup.json"
        if not json_path.exists():
            continue
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            summaries.append(SetupSummary(
                setup_id=metadata.get("setup_id", setup_dir.name),
                part_count=len(metadata.get("parts", [])),
                created_at=metadata.get("created_at", 0.0),
            ))
        except Exception as exc:
            logger.warning("Skipping unreadable setup '%s': %s", setup_dir.name, exc)

    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries
