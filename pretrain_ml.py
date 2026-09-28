from pathlib import Path
from modules.ml_crp import train_and_save

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "database.mat"
OUT = ROOT / "models" / "crp_models.joblib"

if not DB.is_file():
    raise FileNotFoundError(f"Put the training database at: {DB}")

bundle = train_and_save(DB, OUT)
print(f"Saved {len(bundle['models'])} CRP models to {OUT}")
