from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.io
from xgboost import XGBRegressor

from .common import mat_list


def load_training_from_mat(mat_path: str | Path):
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "input" not in mat or "output" not in mat:
        raise KeyError("Training .mat must contain 'input' and 'output'.")
    X = np.asarray(mat["input"])
    Y = np.asarray(mat["output"])
    X_cols = mat_list(mat["index_input"]) if "index_input" in mat else [f"x{i}" for i in range(X.shape[1])]
    Y_cols = mat_list(mat["index_output"]) if "index_output" in mat else [f"y{i}" for i in range(Y.shape[1])]
    return (
        pd.DataFrame(X, columns=X_cols).apply(pd.to_numeric, errors="coerce"),
        pd.DataFrame(Y, columns=Y_cols).apply(pd.to_numeric, errors="coerce"),
        X_cols,
        Y_cols,
    )


def train_and_save(mat_path: str | Path, output_path: str | Path, hyperparam=None):
    """Admin/offline utility. The web app itself should use inference only."""
    if hyperparam is None:
        hyperparam = {"n_estimators": 1000, "learning_rate": 0.5, "max_depth": 10}
    X_df, Y_df, X_cols, Y_cols = load_training_from_mat(mat_path)
    models = {}
    for target in Y_cols:
        d = pd.concat([X_df[X_cols], Y_df[[target]]], axis=1).dropna()
        if len(d) < 50:
            continue
        model = XGBRegressor(
            n_estimators=int(hyperparam["n_estimators"]),
            learning_rate=float(hyperparam["learning_rate"]),
            max_depth=int(hyperparam["max_depth"]),
            random_state=42,
            verbosity=0,
        )
        model.fit(d[X_cols].to_numpy(), d[target].to_numpy())
        models[target] = model
    payload = {"models": models, "X_cols": X_cols, "Y_cols": list(models.keys())}
    joblib.dump(payload, output_path)
    return payload


def load_bundle(path: str | Path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "models" not in bundle or "X_cols" not in bundle:
        raise ValueError("Invalid CRP model bundle.")
    return bundle


def predict(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X_cols = list(bundle["X_cols"])
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing ML input variables: {missing}")
    X = df[X_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    if np.isnan(X).any():
        raise ValueError("ML input contains NaN.")
    pred = {name: model.predict(X) for name, model in bundle["models"].items()}
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(pred)], axis=1)


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Mc_slip" in df.columns:
        mc = pd.to_numeric(df["Mc_slip"], errors="coerce")
        mask = mc >= 1.0
        if "Rc_slip" in df.columns:
            df.loc[mask, "Rc_slip"] = 0.0
        df.loc[mask, "Mc_slip"] = 1.0
        if "Rc_peak" in df.columns:
            df.loc[mask, "Rc_peak"] = 0.1
        if "Mc_peak" in df.columns:
            df.loc[mask, "Mc_peak"] = 1.0
    for c in [c for c in df.columns if c.startswith("M") or c.startswith("NR_")]:
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(upper=1.0)
    return df
