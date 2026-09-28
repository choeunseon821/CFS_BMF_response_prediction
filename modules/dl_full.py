from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

INPUT_NAMES = ["Cd", "Cw", "Cl", "Ct", "bl", "bd", "KL/r", "nbr", "nbc", "phi_b", "Lb", "Fy", "ds", "Rs"]
OUTPUT_COLUMNS = ["MR_conn_x", "MR_conn_y", "MR_frame_x", "MR_frame_y", "NormR_frame_x", "NormR_frame_y"]


def load_assets(model_path: str | Path, x_scaler_path: str | Path, y_scaler_path: str | Path):
    paths = [Path(model_path), Path(x_scaler_path), Path(y_scaler_path)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing DL asset(s): " + ", ".join(missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(paths[0]), map_location=device)
    model.eval()
    scaler_X = joblib.load(paths[1])
    scalers_Y = joblib.load(paths[2])
    if getattr(scaler_X, "n_features_in_", 14) != 14:
        raise ValueError("X scaler must expect 14 input variables.")
    if len(scalers_Y) != 6:
        raise ValueError("Y scaler bundle must contain six scalers.")
    return model, scaler_X, scalers_Y, device


def predict_array(x_input, model, scaler_X, scalers_Y, device):
    x_input = np.asarray(x_input, dtype=np.float32).reshape(-1, 14)
    x_norm = scaler_X.transform(x_input).astype(np.float32)
    x_tensor = torch.from_numpy(x_norm).float().unsqueeze(-1).to(device)
    with torch.no_grad():
        y_norm = model(x_tensor).permute(0, 2, 1).cpu().numpy()
    n, L, C = y_norm.shape
    if C != 6:
        raise ValueError(f"DL output must have 6 channels, got {y_norm.shape}.")
    y_real = np.zeros_like(y_norm, dtype=np.float32)
    for ch in range(C):
        y_real[:, :, ch] = scalers_Y[ch].inverse_transform(y_norm[:, :, ch].reshape(-1, 1)).reshape(n, L)
    y_real[:, :, 5] = np.clip(y_real[:, :, 5], None, 1.0)
    return y_real


def to_dataframe(prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(prediction, columns=OUTPUT_COLUMNS)


def total_curve(prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Rotation": prediction[:, 0] + prediction[:, 2],
        "NormalizedMoment": prediction[:, 3],
    }).sort_values("Rotation").drop_duplicates("Rotation")


def moment_at_rotation(prediction: np.ndarray, rotation: float = 0.04) -> float:
    df = total_curve(prediction)
    x = df["Rotation"].to_numpy(float)
    y = df["NormalizedMoment"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or rotation < x.min() or rotation > x.max():
        return float("nan")
    return float(np.interp(rotation, x, y))


def classify_failure_mode(prediction: np.ndarray) -> str:
    """Response-based descriptive mode from relative connection/frame normalized peak rotations.

    This is intentionally simple; replace with the exact Eq. (1) rule from the manuscript if desired.
    """
    conn_peak = float(np.nanmax(prediction[:, 1]))
    frame_peak = float(np.nanmax(prediction[:, 3]))
    if not np.isfinite(conn_peak) or not np.isfinite(frame_peak):
        return "Unavailable"
    diff = conn_peak - frame_peak
    if abs(diff) <= 0.05:
        return "Balanced"
    return "Connection-dominant" if diff < 0 else "Frame-dominant"
