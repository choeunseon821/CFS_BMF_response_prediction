from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import scipy.io


def mat_list(idx) -> list[str]:
    idx = np.squeeze(idx)
    out = []
    for v in np.atleast_1d(idx):
        if isinstance(v, np.ndarray):
            v = np.squeeze(v)
            try:
                out.append(str(v[0]))
            except Exception:
                out.append(str(v))
        else:
            out.append(str(v))
    return [s.strip() for s in out]


def resolve_database(data_dir: Path) -> Path | None:
    """Return the first .mat database in data_dir, preferring database.mat."""
    preferred = data_dir / "database.mat"
    if preferred.is_file():
        return preferred
    mats = sorted(data_dir.glob("*.mat"))
    return mats[0] if mats else None


def load_design_table(mat_path: str | Path) -> pd.DataFrame:
    """Load design variables from `var` and names from `index_input` when available."""
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "var" not in mat:
        raise KeyError("Database must contain 'var'.")
    var = np.asarray(mat["var"], dtype=float)
    if var.ndim != 2:
        raise ValueError(f"'var' must be 2-D. Got {var.shape}.")
    if var.shape[0] < var.shape[1]:
        var = var.T

    names = None
    if "index_input" in mat:
        try:
            candidate = mat_list(mat["index_input"])
            if len(candidate) == var.shape[1]:
                names = candidate
        except Exception:
            names = None
    if names is None:
        names = [f"var_{i}" for i in range(var.shape[1])]
    return pd.DataFrame(var, columns=names)


def get_domain(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns is not None else list(df.columns)
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            rows.append({
                "Variable": c,
                "Min": float(s.min()),
                "Max": float(s.max()),
                "Median": float(s.median()),
            })
    return pd.DataFrame(rows)


def validate_domain(values: dict[str, float], domain: pd.DataFrame) -> tuple[bool, list[str]]:
    if domain is None or domain.empty:
        return True, []
    lookup = domain.set_index("Variable")[["Min", "Max"]].to_dict("index")
    errors: list[str] = []
    for name, val in values.items():
        if name not in lookup or not np.isfinite(val):
            continue
        lo, hi = float(lookup[name]["Min"]), float(lookup[name]["Max"])
        if val < lo or val > hi:
            errors.append(f"{name}: {val:g} is outside [{lo:g}, {hi:g}]")
    return len(errors) == 0, errors


def rank_candidates(df: pd.DataFrame, mode: str, top_n: int) -> pd.DataFrame:
    """Add interpretable ranking columns without changing the underlying feasibility filter."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()

    # Existing Module 1 output includes margin, nb, Volume and drift moments.
    margin = pd.to_numeric(out.get("margin", np.nan), errors="coerce")
    nb = pd.to_numeric(out.get("nb", np.nan), errors="coerce")
    vol = pd.to_numeric(out.get("Volume", np.nan), errors="coerce")

    system_m = None
    if "M_SMF(0.04rad)" in out.columns:
        system_m = pd.to_numeric(out["M_SMF(0.04rad)"], errors="coerce")
    elif "M_IMF(0.02rad)" in out.columns:
        system_m = pd.to_numeric(out["M_IMF(0.02rad)"], errors="coerce")

    def minmax(s: pd.Series, higher_better: bool = False) -> pd.Series:
        s = s.astype(float)
        lo, hi = s.min(skipna=True), s.max(skipna=True)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            z = pd.Series(0.0, index=s.index)
        else:
            z = (s - lo) / (hi - lo)
        return 1.0 - z if higher_better else z

    material = minmax(vol) + minmax(nb)
    performance = minmax(system_m, higher_better=True) if system_m is not None else minmax(margin)

    if mode == "Performance":
        out["ranking_score"] = performance
    elif mode == "Material efficiency":
        out["ranking_score"] = material
    else:  # Balanced
        out["ranking_score"] = 0.5 * performance + 0.5 * material

    out = out.sort_values("ranking_score", ascending=True).head(int(top_n)).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out
