"""
api/app.py — FastAPI application factory for BoardInspect.

Creates the FastAPI app, mounts static file serving, registers routers,
and configures exception handlers.

The server is bound to 127.0.0.1 (loopback only) in run.py and Uvicorn
config — never 0.0.0.0. No CORS middleware is added because cross-origin
requests are not needed for a loopback-only app.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from boardinspect.config import Config, DEFAULT_CONFIG
from boardinspect.errors import (
    AlignmentError,
    AspectRatioError,
    BoardInspectError,
    CalibrationError,
    ConfigError,
    DuplicateNameError,
    ImageDecodeError,
    ImageTooLargeError,
    ImageTooSmallError,
    InvalidBoxError,
    StorageError,
)

logger = logging.getLogger(__name__)

# Path to the static/ directory (three levels up from this file:
#   src/boardinspect/api/app.py → src/boardinspect/api/ → src/boardinspect/ → src/ → project root)
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "static"


def create_app(config: Config = DEFAULT_CONFIG) -> FastAPI:
    """Create and configure the BoardInspect FastAPI application.

    Registers all exception handlers, the /health liveness endpoint, and
    mounts the static/ frontend directory at the root path.

    Args:
        config: Config instance used throughout the app lifecycle.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="BoardInspect",
        description="Offline PCB component presence checker",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # Attach the config so routes and services can retrieve it via app.state.
    app.state.config = config

    # ── Exception handlers ───────────────────────────────────────────────────
    # Each handler returns a typed JSON error body: {"error": ..., "detail": ...}
    # "error" is user-facing; "detail" is for logs (may be None).

    @app.exception_handler(ImageDecodeError)
    async def handle_image_decode(request: Request, exc: ImageDecodeError) -> JSONResponse:
        logger.debug("ImageDecodeError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=422,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(ImageTooLargeError)
    async def handle_too_large(request: Request, exc: ImageTooLargeError) -> JSONResponse:
        logger.debug("ImageTooLargeError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=413,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(ImageTooSmallError)
    async def handle_too_small(request: Request, exc: ImageTooSmallError) -> JSONResponse:
        logger.debug("ImageTooSmallError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=422,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(AspectRatioError)
    async def handle_aspect_ratio(request: Request, exc: AspectRatioError) -> JSONResponse:
        logger.debug("AspectRatioError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=422,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(DuplicateNameError)
    async def handle_duplicate_name(request: Request, exc: DuplicateNameError) -> JSONResponse:
        logger.debug("DuplicateNameError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(InvalidBoxError)
    async def handle_invalid_box(request: Request, exc: InvalidBoxError) -> JSONResponse:
        logger.debug("InvalidBoxError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(AlignmentError)
    async def handle_alignment(request: Request, exc: AlignmentError) -> JSONResponse:
        logger.debug("AlignmentError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=422,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(StorageError)
    async def handle_storage(request: Request, exc: StorageError) -> JSONResponse:
        logger.warning("StorageError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=500,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(CalibrationError)
    async def handle_calibration(request: Request, exc: CalibrationError) -> JSONResponse:
        logger.debug("CalibrationError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=422,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(ConfigError)
    async def handle_config_error(request: Request, exc: ConfigError) -> JSONResponse:
        logger.debug("ConfigError: %s | detail: %s", exc.message, exc.detail)
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(BoardInspectError)
    async def handle_generic(request: Request, exc: BoardInspectError) -> JSONResponse:
        # Catch-all for any BoardInspectError subclass not handled above.
        logger.error("Unhandled BoardInspectError: %s", exc.message, exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": exc.message, "detail": exc.detail},
        )

    # ── Health endpoint ──────────────────────────────────────────────────────

    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        """Liveness check. Returns {"status": "ok"} when the server is running."""
        return {"status": "ok"}

    # ── API routes ───────────────────────────────────────────────────────────
    from boardinspect.api.routes import router
    app.include_router(router)

    # ── Static files (frontend) ──────────────────────────────────────────────
    # Mounted last so API routes registered above take priority.
    # html=True means index.html is served for directory requests (SPA behaviour).
    if _STATIC_DIR.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(_STATIC_DIR), html=True),
            name="static",
        )
        logger.debug("Mounted static files from %s", _STATIC_DIR)
    else:
        logger.warning(
            "Static directory not found at %s; frontend will not be served.",
            _STATIC_DIR,
        )

    return app
