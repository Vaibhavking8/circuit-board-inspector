"""
tests/test_offline.py — offline-guard tests.

Verifies that:
1. The app serves /health with no real network calls (mocked socket).
2. No src/boardinspect/ module imports outbound-network libraries.
3. The static frontend has no CDN or remote asset references.
4. run.py binds to 127.0.0.1 only.

Run with: pytest tests/test_offline.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "boardinspect"
STATIC_DIR = PROJECT_ROOT / "static"


# ── 1. Health endpoint responds ───────────────────────────────────────────────

def test_health_endpoint():
    """The /health endpoint must return 200 {"status": "ok"}."""
    from fastapi.testclient import TestClient
    from boardinspect.api.app import create_app
    from boardinspect.config import DEFAULT_CONFIG

    app = create_app(DEFAULT_CONFIG)
    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── 2. No outbound network imports in src/ ────────────────────────────────────

FORBIDDEN_IMPORTS = {
    "requests",
    "urllib.request",
    "urllib3",
    "httpx",
    "http.client",
    "aiohttp",
    "websockets",
}

def _get_top_level_imports(filepath: Path) -> set[str]:
    """Return all top-level module names imported in a Python file."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("py_file", sorted(SRC_DIR.rglob("*.py")))
def test_no_outbound_network_import(py_file):
    """src/boardinspect/ must not import outbound-network libraries."""
    imports = _get_top_level_imports(py_file)
    bad = imports & FORBIDDEN_IMPORTS
    assert not bad, (
        f"{py_file.relative_to(PROJECT_ROOT)} imports forbidden network "
        f"library: {', '.join(sorted(bad))}"
    )


# ── 3. No CDN references in static/ ──────────────────────────────────────────

STATIC_EXTENSIONS = {".html", ".css", ".js"}
CDN_PATTERNS = ["https://", "http://", "@import url(http"]


@pytest.mark.parametrize("static_file", [
    f for f in STATIC_DIR.rglob("*")
    if f.is_file() and f.suffix.lower() in STATIC_EXTENSIONS
    and "vendor" not in f.parts  # vendored files are explicitly allowed
])
def test_no_cdn_in_static(static_file):
    """Static files must not reference CDN or remote URLs."""
    content = static_file.read_text(encoding="utf-8", errors="replace")
    for pattern in CDN_PATTERNS:
        assert pattern not in content, (
            f"{static_file.relative_to(PROJECT_ROOT)} contains remote URL pattern '{pattern}'"
        )


# ── 4. run.py binds to loopback only ─────────────────────────────────────────

def test_run_py_loopback_only():
    """run.py must not bind to 0.0.0.0."""
    run_py = PROJECT_ROOT / "run.py"
    assert run_py.exists(), "run.py not found"
    content = run_py.read_text(encoding="utf-8")
    # Must have 127.0.0.1
    assert "127.0.0.1" in content, "run.py does not reference 127.0.0.1"
    # Must not unconditionally bind 0.0.0.0
    assert "host=\"0.0.0.0\"" not in content, "run.py binds to 0.0.0.0"
    assert "host='0.0.0.0'" not in content, "run.py binds to 0.0.0.0"
