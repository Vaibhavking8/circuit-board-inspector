"""Typed exceptions for every BoardInspect failure mode.

The core CV and application layers raise one of these typed errors instead of
returning a sentinel or printing a message. The API layer catches the base
type, logs the technical ``detail`` if present, and turns the user-facing
``message`` into a clear, non-technical response (R9.1). No error derived from
this module may resolve to an OK status (R5.4, R9.2): a failure is always a
"Could not verify" or a per-part Uncertain, never a silent pass.

Each error carries:

``message``
    A non-technical, user-facing explanation (plain language for the shop
    floor).
``detail``
    Optional technical context, intended for log entries only and never shown
    directly to the user.

The hierarchy is intentionally shallow so callers can catch at the right
level::

    BoardInspectError
    +-- ImageDecodeError            (input)
    +-- ImageTooSmallError          (input)
    +-- ImageTooLargeError           (input)
    +-- AspectRatioError             (input)
    +-- DuplicateNameError           (annotation)
    +-- InvalidBoxError              (annotation)
    +-- AlignmentError               (alignment, base)
    |   +-- NonConvergenceError
    |   +-- LowTextureError
    |   +-- RepetitivePatternError
    +-- BorderError                  (border)
    +-- ConfigError                  (configuration)
    +-- StorageError                 (storage)
    +-- CalibrationError             (calibration)

Catching ``BoardInspectError`` catches every BoardInspect failure. Catching
``AlignmentError`` catches every alignment-specific failure without catching
unrelated errors.
"""

from __future__ import annotations

__all__ = [
    "BoardInspectError",
    "ImageDecodeError",
    "ImageTooSmallError",
    "ImageTooLargeError",
    "AspectRatioError",
    "DuplicateNameError",
    "InvalidBoxError",
    "AlignmentError",
    "NonConvergenceError",
    "LowTextureError",
    "RepetitivePatternError",
    "BorderError",
    "ConfigError",
    "StorageError",
    "CalibrationError",
]


class BoardInspectError(Exception):
    """Base class for every BoardInspect error.

    All typed errors raised by the core CV and application layers derive from
    this class, so the API layer can catch a single base type, log the
    technical ``detail`` if present, and surface the user-facing ``message``
    (R9.1). No error derived from this class may resolve to an OK status.

    Attributes:
        message: A non-technical, user-facing explanation.
        detail: Optional technical context, for log entries only.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)


# --- Input errors ---------------------------------------------------------


class ImageDecodeError(BoardInspectError):
    """Raised when an uploaded file cannot be decoded as an image.

    Covers both reference and inspection uploads: bytes that are not a valid
    JPEG/PNG, truncated files, or anything ``cv2.imdecode`` rejects. The file
    extension is never trusted. The system shows a clear error and keeps the
    previous reference (R1.2) or marks that one inspection file as failed
    while continuing with the others (R4.5).
    """


class ImageTooSmallError(BoardInspectError):
    """Raised when an image is smaller than the minimum usable size.

    Below the configured minimum shorter side (default 320 px) detection
    becomes unreliable. The system warns the user rather than proceeding
    silently (R1.5).
    """


class ImageTooLargeError(BoardInspectError):
    """Raised when an image exceeds the configured size or megapixel limits.

    Files above the configurable limits (default 25 MB or 40 megapixels) are
    rejected with a clear message so the demo laptop is not overwhelmed
    (R4.7).
    """


class AspectRatioError(BoardInspectError):
    """Raised when an inspection image's aspect ratio differs from the reference.

    A clearly different aspect ratio means the photo is not the same board in
    the same pose. The image is marked "Could not verify" with the reason
    rather than forcing an unreliable alignment (R4.3, R4.4).
    """


# --- Annotation errors ----------------------------------------------------


class DuplicateNameError(BoardInspectError):
    """Raised when a part name is empty or already used by another part.

    Names are unique per reference setup. The system rejects an empty or
    duplicate name and explains why (R2.2).
    """


class InvalidBoxError(BoardInspectError):
    """Raised when a part box is too small or lies partly outside the image.

    A box smaller than the minimum size (default 8 x 8 px in the reference
    frame) or extending beyond the image bounds is rejected or clipped and the
    user is told which (R2.3).
    """


# --- Alignment errors -----------------------------------------------------


class AlignmentError(BoardInspectError):
    """Base class for alignment-specific failures.

    Catching ``AlignmentError`` catches every alignment failure without
    catching unrelated BoardInspect errors. When all alignment strategies
    fail or the quality is below the configured minimum, the pipeline marks
    the image "Could not verify" rather than guessing (R5.3, R5.4).
    """


class NonConvergenceError(AlignmentError):
    """Raised when an iterative aligner (e.g. ECC) fails to converge.

    The registration cascade catches this and tries the next strategy; if
    every strategy fails the image is marked "Could not verify" (R5.3, R5.4).
    """


class LowTextureError(AlignmentError):
    """Raised when a feature-based aligner finds too few features or matches.

    Textureless regions give ORB too few keypoints to estimate a reliable
    transform. The cascade falls through to the next strategy or to "Could not
    verify" (R5.3, R5.4).
    """


class RepetitivePatternError(AlignmentError):
    """Raised when repetitive texture produces ambiguous feature matches.

    Regular PCB layouts can confuse feature matching. The cascade treats
    this as an alignment failure and falls through (R5.3, R5.4).
    """


# --- Border errors --------------------------------------------------------


class BorderError(BoardInspectError):
    """Raised when a part's aligned box is near or outside the image border.

    Such a part cannot be checked reliably, so the decision engine marks it
    Uncertain with a reason rather than reporting it Missing (R6.5).
    """


# --- Other errors ---------------------------------------------------------


class ConfigError(BoardInspectError):
    """Raised when configuration is invalid (bad value, missing field, etc.).

    All tunables live in one config object. A malformed override or an
    out-of-range value raises this rather than letting an algorithm run with a
    bad parameter (R6.7).
    """


class StorageError(BoardInspectError):
    """Raised when a saved setup is missing files or is corrupt.

    The system shows an error and does not crash (R3.3).
    """


class CalibrationError(BoardInspectError):
    """Raised when per-part calibration cannot be completed.

    If calibration cannot produce a usable threshold for a part, the system
    flags the part as unreliable rather than silently using a guessed value
    (R7.1, R7.4).
    """
