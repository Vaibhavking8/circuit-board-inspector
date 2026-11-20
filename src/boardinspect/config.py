"""
config.py — single source of truth for all BoardInspect tunables.

All thresholds, window sizes and weights live here. Algorithm code must
read from a Config instance; no magic numbers elsewhere.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RegistrationConfig:
    """Parameters for the image alignment (registration) stage."""
    longest_side: int = 1200
    """Resize the image so its longest side is this many pixels before aligning. [EXPERIMENT]"""
    cascade_order: list[str] = field(
        default_factory=lambda: [
            "skip_if_aligned",
            "ecc_euclidean",
            "ecc_affine",
            "orb_homography",
        ]
    )
    """Order in which alignment strategies are tried."""
    ecc_max_iterations: int = 50
    """Maximum ECC iterations before declaring non-convergence."""
    ecc_epsilon: float = 1e-3
    """ECC convergence threshold (terminates when delta < epsilon)."""
    min_alignment_quality: float = 0.5
    """Minimum quality score [0, 1] to accept an alignment result."""
    transform_max_translation_fraction: float = 0.3
    """Reject transforms that translate more than this fraction of image size."""
    transform_min_scale: float = 0.8
    """Reject transforms with scale below this."""
    transform_max_scale: float = 1.2
    """Reject transforms with scale above this."""
    transform_max_rotation_degrees: float = 10.0
    """Reject transforms that rotate more than this many degrees."""
    orb_max_features: int = 500
    """Maximum ORB keypoints to detect."""
    orb_min_matches: int = 10
    """Minimum feature matches required to attempt homography estimation."""
    orb_ransac_threshold: float = 5.0
    """RANSAC reprojection threshold in pixels."""
    skip_if_aligned_tolerance_px: float = 1.0
    """If mean corner displacement is below this, skip further alignment."""


@dataclass
class ComparisonConfig:
    """Parameters for the per-part comparison stage."""
    longest_side: int = 2400
    """Resize the image so its longest side is this many pixels for comparison. [EXPERIMENT]"""
    search_window_padding_fraction: float = 0.5
    """Pad the search window by this fraction of the box dimensions on each side."""
    search_window_neighbour_limit_fraction: float = 0.5
    """Cap the search window half-width at this fraction of the nearest-neighbour distance."""
    signals_enabled: list[str] = field(default_factory=lambda: ["template", "edge"])
    """Signals to compute and combine for the presence score."""
    signal_weights: dict[str, float] = field(
        default_factory=lambda: {"template": 1.0, "edge": 1.0}
    )
    """Weight for each signal when computing the combined score."""
    template_method: str = "TM_CCOEFF_NORMED"
    """OpenCV matchTemplate method name (without cv2. prefix)."""
    subpixel_refinement: bool = False
    """Apply parabolic sub-pixel refinement to the template match peak. [EXPERIMENT]"""


@dataclass
class PreprocessingConfig:
    """Image preprocessing options applied before registration and comparison."""
    grayscale: bool = True
    """Convert to grayscale before processing."""
    gaussian_blur: bool = False
    """Apply a mild Gaussian blur before processing."""
    gaussian_kernel_size: int = 3
    """Kernel size for the Gaussian blur (must be odd)."""
    clahe: bool = False
    """Apply CLAHE histogram equalisation. [EXPERIMENT]"""
    clahe_clip_limit: float = 2.0
    """CLAHE clip limit."""
    clahe_grid_size: int = 8
    """CLAHE tile grid size (tiles are square)."""


@dataclass
class DecisionConfig:
    """Parameters for the per-part presence/absence decision."""
    uncertain_margin: float = 0.1
    """Scores within this margin of the threshold are marked Uncertain."""
    border_fraction: float = 0.05
    """Parts whose aligned box falls within this fraction of the image border → Uncertain."""
    sensitivity_min: float = 0.0
    """Minimum user-facing sensitivity (most lenient)."""
    sensitivity_max: float = 1.0
    """Maximum user-facing sensitivity (most strict)."""
    sensitivity_default: float = 0.5
    """Default sensitivity value."""
    threshold_method: str = "midpoint"
    """How to place the decision threshold between the present and absent score distributions."""


@dataclass
class CalibrationConfig:
    """Parameters for synthetic-augmentation calibration."""
    sample_count: int = 20
    """Number of augmented samples generated per part per condition (present/absent)."""
    shift_px: int = 5
    """Maximum pixel shift in each direction during augmentation."""
    rotation_degrees: float = 3.0
    """Maximum rotation in degrees during augmentation."""
    brightness: float = 0.20
    """Brightness variation as a fraction (±20 %)."""
    contrast: float = 0.20
    """Contrast variation as a fraction (±20 %)."""
    noise_std: float = 5.0
    """Standard deviation of additive Gaussian noise."""
    blur_kernel: int = 3
    """Mild blur kernel applied to some augmented samples."""
    threshold_margin: float = 0.05
    """Required separation margin between present/absent score distributions."""


@dataclass
class ImageLimitsConfig:
    """Hard limits on uploaded images."""
    max_file_size_mb: float = 25.0
    """Maximum upload file size in megabytes."""
    max_megapixels: float = 40.0
    """Maximum image area in megapixels."""
    recommended_megapixels: float = 12.0
    """Recommended maximum area; warn if exceeded."""
    min_shorter_side: int = 240
    """Minimum length of the shorter image dimension in pixels."""


@dataclass
class PartLimitsConfig:
    """Hard limits on part annotations."""
    max_parts: int = 30
    """Maximum number of named parts per reference setup."""
    min_box_size_px: int = 8
    """Minimum part box side length in reference-image pixels."""


@dataclass
class Config:
    """
    Root configuration object for BoardInspect.

    Instantiate directly for defaults or use load_config() to apply JSON overrides.
    All algorithm code reads parameters from this object; no magic numbers in src/.
    """
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    image_limits: ImageLimitsConfig = field(default_factory=ImageLimitsConfig)
    part_limits: PartLimitsConfig = field(default_factory=PartLimitsConfig)
    random_seed: int = 42
    """Fixed random seed for deterministic output."""


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Config = Config()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def config_to_dict(config: Config) -> dict[str, Any]:
    """Convert a Config instance to a plain JSON-serialisable dict."""
    return asdict(config)


def _merge_into_dataclass(instance: Any, overrides: dict[str, Any]) -> Any:
    """Recursively apply *overrides* onto a dataclass *instance* and return a new instance."""
    import dataclasses

    if not dataclasses.is_dataclass(instance):
        return overrides  # leaf value: use the override directly

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(instance):
        if f.name in overrides:
            child_value = overrides[f.name]
            child_instance = getattr(instance, f.name)
            if dataclasses.is_dataclass(child_instance) and isinstance(child_value, dict):
                kwargs[f.name] = _merge_into_dataclass(child_instance, child_value)
            else:
                kwargs[f.name] = child_value
        else:
            kwargs[f.name] = getattr(instance, f.name)
    return type(instance)(**kwargs)


def from_dict(data: dict[str, Any]) -> Config:
    """Reconstruct a Config from a plain dict (e.g. loaded from JSON)."""
    return _merge_into_dataclass(Config(), data)


def load_config(path: str | Path | None = None) -> Config:
    """
    Load a Config, optionally overriding defaults from a JSON file at *path*.

    If *path* is None, returns DEFAULT_CONFIG unchanged.
    Unknown keys in the JSON are silently ignored so old snapshots stay loadable.
    """
    if path is None:
        return DEFAULT_CONFIG

    path = Path(path)
    if not path.exists():
        logger.warning("Config file %s not found; using defaults.", path)
        return DEFAULT_CONFIG

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    config = _merge_into_dataclass(Config(), raw)
    logger.debug("Loaded config from %s", path)
    return config
