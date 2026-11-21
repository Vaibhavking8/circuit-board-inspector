"""
api/services/setup_service.py — thin adapter for setup save/load.

Delegates to storage.save_setup / load_setup / list_setups. No CV logic.
"""
from __future__ import annotations

import logging

import numpy as np

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.models import ReferenceSetup
from boardinspect.storage import SetupSummary, list_setups, load_setup, save_setup

logger = logging.getLogger(__name__)


def save(
    setup: ReferenceSetup,
    image_bytes: bytes,
    config: Config = DEFAULT_CONFIG,
) -> str:
    """Save a reference setup and return the setup_id."""
    setup_id = save_setup(setup, image_bytes, config)
    logger.debug("Saved setup %s", setup_id)
    return setup_id


def load(
    setup_id: str,
    config: Config = DEFAULT_CONFIG,
) -> tuple[ReferenceSetup, np.ndarray]:
    """Load a saved reference setup. Raises StorageError on failure."""
    setup, image = load_setup(setup_id, config)
    logger.debug("Loaded setup %s with %d parts", setup_id, len(setup.parts))
    return setup, image


def list_all(config: Config = DEFAULT_CONFIG) -> list[SetupSummary]:
    """List all saved setups, newest first."""
    return list_setups(config)
