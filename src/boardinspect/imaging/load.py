"""
imaging/load.py — image decoding, EXIF orientation, and upload validation.

All functions are pure: they take bytes or arrays and return arrays or
typed results. No file paths are opened here (callers pass bytes).
No web-framework, storage, or network imports.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

import cv2
import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.errors import ImageDecodeError, ImageTooLargeError, ImageTooSmallError

logger = logging.getLogger(__name__)


@dataclass
class ValidatedImage:
    """Result of a successful image validation."""
    array: np.ndarray          # BGR image, possibly orientation-corrected
    original_width: int        # width AFTER orientation correction
    original_height: int       # height AFTER orientation correction
    file_size_bytes: int       # size of the raw bytes


def _exif_orientation(raw_bytes: bytes) -> int:
    """Extract EXIF orientation tag from JPEG bytes, or return 1 (normal).

    Reads the EXIF IFD directly from bytes without external libraries.
    Returns an integer 1–8 matching the EXIF Orientation tag values:
      1 = normal, 3 = 180°, 6 = 90° CW, 8 = 90° CCW.
    Returns 1 for non-JPEG or missing/corrupt EXIF.
    """
    # Only JPEG has EXIF in-band; check the SOI marker
    if len(raw_bytes) < 4 or raw_bytes[:2] != b'\xff\xd8':
        return 1

    offset = 2
    while offset + 4 <= len(raw_bytes):
        marker = raw_bytes[offset:offset + 2]
        if len(marker) < 2:
            break
        seg_len_bytes = raw_bytes[offset + 2:offset + 4]
        if len(seg_len_bytes) < 2:
            break
        seg_len = struct.unpack('>H', seg_len_bytes)[0]

        if marker == b'\xff\xe1':  # APP1 (EXIF)
            app1 = raw_bytes[offset + 4: offset + 2 + seg_len]
            if app1[:6] == b'Exif\x00\x00':
                tiff = app1[6:]
                if len(tiff) < 8:
                    return 1
                byte_order = tiff[:2]
                if byte_order == b'II':
                    endian = '<'
                elif byte_order == b'MM':
                    endian = '>'
                else:
                    return 1
                ifd_offset = struct.unpack(endian + 'I', tiff[4:8])[0]
                if ifd_offset + 2 > len(tiff):
                    return 1
                n_entries = struct.unpack(endian + 'H', tiff[ifd_offset:ifd_offset + 2])[0]
                for i in range(n_entries):
                    entry_start = ifd_offset + 2 + i * 12
                    if entry_start + 12 > len(tiff):
                        break
                    tag = struct.unpack(endian + 'H', tiff[entry_start:entry_start + 2])[0]
                    if tag == 0x0112:  # Orientation
                        value = struct.unpack(endian + 'H', tiff[entry_start + 8:entry_start + 10])[0]
                        return value
        elif marker[0:1] == b'\xff' and marker[1:2] in (b'\xd9', b'\xda'):
            break  # EOI or SOS — stop scanning

        offset += 2 + seg_len

    return 1


def _apply_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate/flip *image* (BGR HxWxC) to correct for EXIF orientation."""
    if orientation == 1:
        return image
    elif orientation == 2:
        return cv2.flip(image, 1)
    elif orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        return cv2.flip(image, 0)
    elif orientation == 5:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1)
    elif orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        return cv2.flip(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    elif orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        logger.warning("Unknown EXIF orientation %d; leaving image unchanged.", orientation)
        return image


def load_image(raw_bytes: bytes) -> np.ndarray:
    """Decode *raw_bytes* as a BGR image with EXIF orientation applied.

    Args:
        raw_bytes: Raw file bytes (JPEG, PNG, BMP, TIFF, WebP, etc.).

    Returns:
        BGR uint8 ndarray with EXIF orientation corrected.

    Raises:
        ImageDecodeError: If the bytes cannot be decoded as an image.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError(
            "The file could not be read as an image. "
            "Please upload a valid JPEG, PNG, BMP, TIFF, or WebP file.",
            detail=f"cv2.imdecode returned None for {len(raw_bytes)} bytes",
        )

    orientation = _exif_orientation(raw_bytes)
    if orientation != 1:
        logger.debug("Applying EXIF orientation %d.", orientation)
        image = _apply_orientation(image, orientation)

    return image


def validate_image(
    raw_bytes: bytes,
    filename: str = "<upload>",
    config: Config = DEFAULT_CONFIG,
) -> ValidatedImage:
    """Validate *raw_bytes* as a usable image, enforcing config limits.

    Validation order:
    1. File size limit (config.image_limits.max_file_size_mb).
    2. Decode check — extension is ignored, bytes are decoded.
    3. EXIF orientation correction.
    4. Minimum dimension check (config.image_limits.min_shorter_side).
    5. Megapixel limit (config.image_limits.max_megapixels).
    6. Recommended megapixel warning (non-fatal).

    Args:
        raw_bytes: Raw upload bytes.
        filename:  Original filename, used only for log messages.
        config:    Config instance (default: DEFAULT_CONFIG).

    Returns:
        ValidatedImage with the decoded, orientation-corrected array.

    Raises:
        ImageTooLargeError: File size or megapixel limit exceeded.
        ImageTooSmallError: Image is smaller than the minimum usable size.
        ImageDecodeError:   Bytes cannot be decoded as an image.
    """
    limits = config.image_limits
    file_size_bytes = len(raw_bytes)

    # 1. File size
    max_bytes = limits.max_file_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise ImageTooLargeError(
            f"File is too large ({file_size_bytes / 1_048_576:.1f} MB). "
            f"Maximum allowed size is {limits.max_file_size_mb:.0f} MB.",
            detail=f"{filename}: {file_size_bytes} bytes > {max_bytes:.0f} bytes",
        )

    # 2 & 3. Decode and apply EXIF orientation
    image = load_image(raw_bytes)
    h, w = image.shape[:2]

    # 4. Minimum dimension
    shorter = min(h, w)
    if shorter < limits.min_shorter_side:
        raise ImageTooSmallError(
            f"Image is too small ({w}×{h} px). "
            f"The shorter side must be at least {limits.min_shorter_side} px.",
            detail=f"{filename}: shorter side {shorter} px < {limits.min_shorter_side} px",
        )

    # 5. Megapixel limit
    megapixels = (h * w) / 1_000_000
    if megapixels > limits.max_megapixels:
        raise ImageTooLargeError(
            f"Image is too large ({megapixels:.1f} MP). "
            f"Maximum allowed size is {limits.max_megapixels:.0f} MP.",
            detail=f"{filename}: {megapixels:.2f} MP > {limits.max_megapixels} MP",
        )

    # 6. Recommended megapixel warning (non-fatal)
    if megapixels > limits.recommended_megapixels:
        logger.warning(
            "%s: image is %s MP (recommended ≤ %s MP). Processing may be slow.",
            filename, f"{megapixels:.1f}", f"{limits.recommended_megapixels:.0f}",
        )

    logger.debug("Validated %s: %d×%d px, %.2f MP, %d bytes.", filename, w, h, megapixels, file_size_bytes)

    return ValidatedImage(
        array=image,
        original_width=w,
        original_height=h,
        file_size_bytes=file_size_bytes,
    )
