"""
tools/verify_install.py — verify the BoardInspect installation is complete.

Run this after installing to confirm everything works before the demo.

Usage:
    python tools/verify_install.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REQUIRED_PACKAGES = [
    ("cv2",             "opencv-python-headless"),
    ("numpy",           "numpy"),
    ("fastapi",         "fastapi"),
    ("uvicorn",         "uvicorn"),
    ("multipart",       "python-multipart"),
]

REQUIRED_MODULES = [
    "boardinspect.config",
    "boardinspect.errors",
    "boardinspect.models",
    "boardinspect.coordinates",
    "boardinspect.imaging.load",
    "boardinspect.imaging.preprocess",
    "boardinspect.registration.cascade",
    "boardinspect.comparison.template",
    "boardinspect.comparison.edge",
    "boardinspect.decision.engine",
    "boardinspect.decision.calibration",
    "boardinspect.pipeline",
    "boardinspect.storage",
    "boardinspect.viz",
    "boardinspect.api.app",
]


def check_packages() -> bool:
    ok = True
    for mod_name, pkg_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(mod_name)
            print(f"  \u2713 {pkg_name}")
        except ImportError:
            print(f"  \u2717 {pkg_name} \u2014 NOT FOUND (run: pip install -r requirements.txt)")
            ok = False
    return ok


def check_modules() -> bool:
    ok = True
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
            print(f"  \u2713 {mod}")
        except Exception as exc:
            print(f"  \u2717 {mod} \u2014 {exc}")
            ok = False
    return ok


def check_static() -> bool:
    static = Path(__file__).resolve().parent.parent / "static"
    missing = []
    for path in [
        "index.html",
        "css/styles.css",
        "js/main.js",
        "js/state.js",
        "js/api-client.js",
        "js/coordinate-utils.js",
    ]:
        if not (static / path).exists():
            missing.append(path)
    if missing:
        for m in missing:
            print(f"  \u2717 static/{m} \u2014 missing")
        return False
    print("  \u2713 static/ frontend files present")
    return True


def check_data_dirs() -> bool:
    root = Path(__file__).resolve().parent.parent
    dirs = ["data/samples", "data/runtime"]
    ok = True
    for d in dirs:
        if not (root / d).exists():
            print(f"  \u2717 {d} \u2014 missing (run: mkdir -p {d})")
            ok = False
        else:
            print(f"  \u2713 {d}/")
    return ok


def check_api() -> bool:
    try:
        from fastapi.testclient import TestClient
        from boardinspect.api.app import create_app
        from boardinspect.config import DEFAULT_CONFIG

        app = create_app(DEFAULT_CONFIG)
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        print("  \u2713 FastAPI app starts and /health responds")
        return True
    except Exception as exc:
        print(f"  \u2717 FastAPI app check failed: {exc}")
        return False


def main() -> None:
    print("\n=== BoardInspect Installation Verification ===\n")

    results: dict[str, bool] = {}

    print("1. Required packages:")
    results["packages"] = check_packages()

    print("\n2. boardinspect modules:")
    results["modules"] = check_modules()

    print("\n3. Frontend static files:")
    results["static"] = check_static()

    print("\n4. Data directories:")
    results["data"] = check_data_dirs()

    print("\n5. API health check:")
    results["api"] = check_api()

    print()
    if all(results.values()):
        print("\u2705  All checks passed. BoardInspect is ready to run.")
        print("    Start with: python run.py")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\u274c  {len(failed)} check(s) failed: {', '.join(failed)}")
        print("    Fix the issues above, then re-run: python tools/verify_install.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
