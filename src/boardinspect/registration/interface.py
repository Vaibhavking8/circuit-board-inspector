"""
registration/interface.py — interface contract for image alignment strategies.

Each registration strategy (ECC, ORB, identity) implements the Registrar
protocol. The cascade runner in cascade.py works against this interface,
so strategies can be swapped or added by adding a module.

No web-framework, storage, or imaging imports here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.models import TransformType


class RegistrationStatus(str, Enum):
    """Outcome of a single registration attempt."""
    Aligned = "aligned"     # transform found and quality above minimum
    Failed = "failed"       # all strategies exhausted or quality too low
    Identity = "identity"   # skip-if-aligned: images are already aligned


@dataclass
class RegistrationResult:
    """Result returned by any Registrar implementation.

    Attributes:
        transform:      2×3 affine matrix (float32) or 3×3 homography (float32).
                        For Identity status, this is the 2×3 identity matrix.
        quality:        Alignment quality in [0, 1]. Higher is better.
                        0.0 for Failed; 1.0 for Identity.
        method:         Name of the strategy that produced this result
                        (e.g. "ecc_euclidean", "orb_homography", "identity").
        status:         Aligned / Failed / Identity.
        transform_type: Geometric model of the transform.
    """
    transform: np.ndarray
    quality: float
    method: str
    status: RegistrationStatus
    transform_type: TransformType

    @classmethod
    def identity(cls) -> "RegistrationResult":
        """Return a canonical identity result (images already aligned)."""
        return cls(
            transform=np.eye(2, 3, dtype=np.float32),
            quality=1.0,
            method="identity",
            status=RegistrationStatus.Identity,
            transform_type=TransformType.Identity,
        )

    @classmethod
    def failed(cls, method: str = "cascade") -> "RegistrationResult":
        """Return a canonical failure result."""
        return cls(
            transform=np.eye(2, 3, dtype=np.float32),
            quality=0.0,
            method=method,
            status=RegistrationStatus.Failed,
            transform_type=TransformType.Identity,
        )


@runtime_checkable
class Registrar(Protocol):
    """Protocol that every alignment strategy must satisfy.

    A Registrar takes a reference grayscale image and a test grayscale image
    and returns a RegistrationResult. It must not perform any file I/O or
    network calls. All parameters come from the Config object.
    """

    def register(
        self,
        ref_gray: np.ndarray,
        test_gray: np.ndarray,
        config: Config,
    ) -> RegistrationResult:
        """Align *test_gray* to *ref_gray*.

        Args:
            ref_gray:  Reference image, single-channel uint8 or float32.
            test_gray: Test image, single-channel uint8 or float32.
            config:    Config instance with registration parameters.

        Returns:
            RegistrationResult with the best transform found.
            If alignment fails, return RegistrationResult.failed() rather
            than raising — the cascade handles failure by trying the next
            strategy.

        Note:
            Strategies that encounter a *typed* error condition (e.g.
            NonConvergenceError, LowTextureError) may raise those errors;
            the cascade catches them and tries the next strategy.
        """
        ...
