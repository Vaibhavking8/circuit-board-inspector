"""
comparison/interface.py — interface contract for per-part comparison signals.

Each signal (template match, edge/gradient, optional SSIM) implements the
Comparator protocol. The pipeline calls each enabled signal independently,
then the decision engine combines the per-signal scores.

Adding a new signal means adding one module that implements Comparator; no
existing callers need to change.

No web-framework, storage, or imaging imports here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG


@dataclass
class SignalResult:
    """Score produced by one comparison signal for one part.

    Attributes:
        name:     Signal identifier (e.g. ``"template"``, ``"edge"``).
        value:    Normalised score in [-1, 1] or [0, 1] depending on the
                  signal. Higher means more likely Present.
        location: Best-match offset (dx, dy) in pixels within the search
                  window. ``None`` if the signal does not produce a location
                  (e.g. a global histogram signal).
    """
    name: str
    value: float
    location: tuple[int, int] | None = None


@runtime_checkable
class Comparator(Protocol):
    """Protocol that every comparison signal must satisfy.

    A Comparator receives a reference crop (the part as it appears on the
    reference board) and a search region (a padded area on the aligned
    inspection image centred on the expected part location). It returns a
    SignalResult with a score and optionally the best-match offset.

    Implementations must be pure: no file I/O, no network, no web-framework
    imports. All parameters come from the Config object.
    """

    @property
    def name(self) -> str:
        """Identifier matching an entry in config.comparison.signals_enabled."""
        ...

    def compare(
        self,
        ref_crop: np.ndarray,
        search_region: np.ndarray,
        config: Config,
    ) -> SignalResult:
        """Compare *ref_crop* against *search_region*.

        Args:
            ref_crop:      Grayscale crop of the part from the reference image
                           at comparison resolution (uint8, H×W).
            search_region: Grayscale padded search window from the inspection
                           image at comparison resolution (uint8, H'×W').
                           H' ≥ H and W' ≥ W (padded by search_window config).
            config:        Config instance with comparison parameters.

        Returns:
            SignalResult for this part at this inspection image.
        """
        ...
