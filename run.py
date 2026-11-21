"""
run.py — start the BoardInspect local server and open the default browser.

Usage:
    python run.py [--host 127.0.0.1] [--port 8000]

The server listens on the loopback address only (127.0.0.1) for security.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure the src/ directory is on the path so boardinspect is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn

from boardinspect.api.app import create_app
from boardinspect.config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger("run")


def _open_browser(url: str, delay: float = 1.0) -> None:
    """Wait *delay* seconds then open *url* in the default browser."""
    def _target() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Could not open browser: %s", exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the BoardInspect server.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1). Security: not 0.0.0.0.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only).")
    args = parser.parse_args()

    # Enforce loopback-only binding (R10.6)
    if args.host not in ("127.0.0.1", "localhost"):
        logger.warning(
            "Host '%s' is not the loopback address. Overriding to 127.0.0.1 for security.",
            args.host,
        )
        args.host = "127.0.0.1"

    url = f"http://{args.host}:{args.port}"
    logger.info("Starting BoardInspect at %s", url)

    if not args.no_browser:
        _open_browser(url, delay=1.5)
    else:
        print(f"BoardInspect: open {url} in your browser.")

    app = create_app(DEFAULT_CONFIG)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=False,  # reload=True requires a string app reference; keep simple
        log_level="info",
    )


if __name__ == "__main__":
    main()
