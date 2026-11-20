"""
models.py — dataclass domain models for BoardInspect.

Coordinates are stored normalised in [0, 1] relative to the reference image's
original pixel dimensions. Pixel coordinates are always derived at runtime; they
are never persisted. See coordinates.py for the canonical converter.

Schema versioning: bump SCHEMA_VERSION whenever a field is added or removed so
storage.py can detect incompatible saved setups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PartStatus(str, Enum):
    """Per-part presence decision."""
    Present = "present"
    Missing = "missing"
    Uncertain = "uncertain"


class ImageStatus(str, Enum):
    """Overall verdict for one inspection image."""
    OK = "ok"
    MissingParts = "missing_parts"
    CouldNotVerify = "could_not_verify"


class TransformType(str, Enum):
    """Type of geometric transform produced by registration."""
    Identity = "identity"
    Translation = "translation"
    Euclidean = "euclidean"
    Affine = "affine"
    Homography = "homography"


# ---------------------------------------------------------------------------
# Part annotation
# ---------------------------------------------------------------------------

@dataclass
class Part:
    """A named part region annotated on the reference image.

    ``box`` is normalised (x, y, w, h) in [0, 1] relative to the reference
    image's original dimensions, origin top-left.  ``id`` is a stable
    string identifier (e.g. a UUID or slugged name) that persists across
    saves/loads.  ``name`` is the user-visible label; must be unique within
    a ReferenceSetup.
    """
    id: str
    name: str
    box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        x, y, w, h = self.box
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Part '{self.name}': box origin ({x}, {y}) out of [0, 1].")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise ValueError(f"Part '{self.name}': box size ({w}, {h}) must be in (0, 1].")
        if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
            raise ValueError(
                f"Part '{self.name}': box extends outside image (x+w={x+w:.4f}, y+h={y+h:.4f})."
            )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class SignalStats:
    """Score distribution statistics for one signal under one condition."""
    mean: float
    std: float


@dataclass
class CalibrationResult:
    """Per-part calibration outcome produced by the calibration pipeline.

    ``signal_stats`` maps signal name → condition ('present'/'absent') → stats.
    ``threshold`` is the decision boundary between present and absent.
    ``reliable`` is False when the distributions overlap too much.
    ``reason`` explains why the calibration is unreliable (empty string when reliable).
    """
    signal_stats: dict[str, dict[str, SignalStats]]
    threshold: float
    reliable: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Reference setup
# ---------------------------------------------------------------------------

@dataclass
class ReferenceSetup:
    """Everything that describes a saved reference board configuration.

    ``content_hash`` is a SHA-256 hex digest of the reference image bytes,
    used to detect if the saved image file has been tampered with.
    ``config_snapshot`` is a plain-dict copy of the Config at the time of
    annotation, so the saved setup is self-contained.
    ``schema_version`` must equal SCHEMA_VERSION; storage raises StorageError
    when it doesn't.
    """
    content_hash: str
    original_width: int
    original_height: int
    parts: list[Part]
    config_snapshot: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    calibration: dict[str, CalibrationResult] | None = None


# ---------------------------------------------------------------------------
# Inspection results
# ---------------------------------------------------------------------------

@dataclass
class AlignmentInfo:
    """Summary of the registration step for one inspection image."""
    strategy: str
    transform_type: TransformType
    quality: float


@dataclass
class PartResult:
    """Presence/absence result for one annotated part in one inspection image.

    ``inspection_box`` is the aligned part box in *pixel* coordinates on the
    inspection image — this is derived at runtime and NOT normalised because it
    is consumed directly by the visualisation layer without further scaling.
    ``margin`` is the signed distance from the threshold (positive → more
    present-like, negative → more absent-like).
    ``reason`` is a short human-readable explanation, populated when status is
    Uncertain or when something unusual happened.
    """
    part_id: str
    status: PartStatus
    signals: dict[str, float]
    threshold: float
    margin: float
    inspection_box: tuple[int, int, int, int]
    reason: str | None = None


@dataclass
class ImageResult:
    """Full inspection result for one uploaded image.

    ``status`` is the overall verdict: OK means every part is Present;
    MissingParts means at least one part is Missing (Uncertain parts do not
    trigger MissingParts on their own); CouldNotVerify means alignment failed
    so no per-part decisions could be made.

    A failure NEVER resolves to OK — if alignment fails, status is CouldNotVerify.
    """
    status: ImageStatus
    alignment: AlignmentInfo
    part_results: list[PartResult]
    runtime_ms: float
    messages: list[str] = field(default_factory=list)
