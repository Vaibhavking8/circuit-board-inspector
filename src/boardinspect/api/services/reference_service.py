"""
api/services/reference_service.py — thin adapter for reference upload.

No CV logic here — delegates to imaging.load.validate_image.
"""
from __future__ import annotations

import hashlib
import logging

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.imaging.load import validate_image, ValidatedImage
from boardinspect.models import ReferenceSetup

logger = logging.getLogger(__name__)


def process_upload(
    raw_bytes: bytes,
    filename: str = "<upload>",
    config: Config = DEFAULT_CONFIG,
) -> tuple[str, ValidatedImage, ReferenceSetup]:
    """Validate *raw_bytes* as a reference image and build an initial ReferenceSetup.

    Args:
        raw_bytes: Raw upload bytes.
        filename:  Original filename for logging.
        config:    Config instance.

    Returns:
        Tuple of (reference_id, ValidatedImage, ReferenceSetup with no parts).

    Raises:
        ImageDecodeError, ImageTooLargeError, ImageTooSmallError: on validation failure.
    """
    validated = validate_image(raw_bytes, filename, config)
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    reference_id = content_hash[:16]

    setup = ReferenceSetup(
        content_hash=content_hash,
        original_width=validated.original_width,
        original_height=validated.original_height,
        parts=[],
        config_snapshot={},
    )

    logger.debug(
        "Reference processed: id=%s, %dx%d",
        reference_id,
        validated.original_width,
        validated.original_height,
    )
    return reference_id, validated, setup
