from __future__ import annotations

from pathlib import Path
import importlib.util
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REQ = ROOT / "requirements.txt"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

REQUIRED = {
    "streamlit": "streamlit",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "torch": "torch",
    "openpyxl": "openpyxl",
    "reportlab": "reportlab",
}


def install_missing_packages() -> None:
    missing = [pip_name for module, pip_name in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if not missing:
        print("[OK] Python packages ready.")
        return
    print("[SETUP] Installing missing packages: " + ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQ)])


def _find_exact(name: str) -> Path | None:
    """Look for a required project asset near the project folder.

    This is intentionally limited to a few parent folders so startup does not
    scan an entire drive.
    """
    direct = ROOT / name
    if direct.is_file():
        return direct

    search_roots = [ROOT.parent, ROOT.parent.parent]
    seen = set()
    for base in search_roots:
        try:
            key = str(base.resolve())
        except Exception:
            key = str(base)
        if key in seen or not base.exists():
            continue
        seen.add(key)
        try:
            for p in base.rglob(name):
                if p.is_file() and ROOT not in p.parents:
                    return p
        except (PermissionError, OSError):
            continue
    return None


def prepare_assets() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # Database: keep any .mat already in /data. Otherwise look for the common
    # project database names near the project directory and copy once.
    if not any(DATA_DIR.glob("*.mat")):
        for name in ("database.mat", "dataset_20260831.mat"):
            src = _find_exact(name)
            if src:
                dst = DATA_DIR / src.name
                shutil.copy2(src, dst)
                print(f"[AUTO] Database copied from: {src}")
                break

    # DL assets: scalers are distributed with the package; if the TorchScript
    # model exists elsewhere in the user's research directory, copy it once.
    for name in ("BCRP_script.pt", "BCRP_dataset_X.pkl", "BCRP_dataset_Y.pkl"):
        dst = MODEL_DIR / name
        if dst.is_file():
            continue
        src = _find_exact(name)
        if src:
            shutil.copy2(src, dst)
            print(f"[AUTO] {name} copied from: {src}")


def launch() -> None:
    print("\n============================================================")
    print(" CFS-BMF Design & Response Prediction Platform")
    print("============================================================")
    print(f"Python : {sys.executable}")
    print(f"Project: {ROOT}\n")

    install_missing_packages()
    prepare_assets()

    print("[START] Opening platform at http://localhost:8501")
    print("        Stop with Ctrl+C.\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")],
        cwd=str(ROOT),
        check=False,
    )


if __name__ == "__main__":
    launch()
