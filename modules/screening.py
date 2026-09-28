
"""
Module 1: Combination Recommendation (CFS bolted connection)

Designed to be called from a Streamlit app.

- Reads MATLAB .mat produced by your pipeline (var, Mp, MR_conn/MR_frame/NormR_frame or dataset).
- Builds bolt-connection M–R backbone: x_bolt = MR_conn.x + MR_frame.x, y_bolt = MR_conn.y * Mp
- Computes Mp, M_req=0.8Mp, M_IMF=M(0.02), M_SMF=M(0.04), x_at_M (rotation at applied moment)
- Filters by frame length (optional) and performance requirement (IMF / SMF)
- Scores and returns Top-N combinations.

Notes
- Some .mat files store var/Mp as (nVar, N). This module auto-detects and transposes if needed.
- If Streamlit UI uses meters (1.5/2.0/2.5) but var stores mm (1500/2000/2500), auto-scaling is applied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.io


def MatList(idx) -> list[str]:
    """Convert MATLAB cell/string arrays loaded by scipy.io.loadmat into list of strings."""
    idx = np.squeeze(idx)
    out = []
    for v in idx:
        if isinstance(v, np.ndarray):
            v = np.squeeze(v)
            try:
                out.append(str(v[0]))
            except Exception:
                out.append(str(v))
        else:
            out.append(str(v))
    return [s.strip() for s in out]


def cell_to_matrix(cell) -> np.ndarray:
    """
    Convert MATLAB cell array of vectors into 2D numeric array (L, N).

    Supports object arrays (MATLAB cell) loaded by scipy.io.loadmat.
    """
    arr = np.asarray(cell)
    arr = np.squeeze(arr)

    # Already numeric
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return np.array(arr, dtype=np.float32)

    if not isinstance(arr, np.ndarray) or arr.dtype != object:
        return np.array(arr, dtype=np.float32)

    if arr.ndim == 0:
        v = np.asarray(arr.item()).reshape(-1, 1)
        return v.astype(float)

    if arr.ndim == 1:
        cols = []
        for i in range(arr.shape[0]):
            cols.append(np.asarray(arr[i]).reshape(-1))
        return np.column_stack(cols).astype(float)

    # 2D cell (e.g., 1xN or Nx1)
    flat = arr.reshape(-1)
    cols = []
    for i in range(flat.shape[0]):
        cols.append(np.asarray(flat[i]).reshape(-1))
    return np.column_stack(cols).astype(float)


def interp_x_at_y(M: float, x: np.ndarray, y: np.ndarray) -> float:
    """Return rotation x at moment y=M using linear interpolation; NaN if out of range."""
    if not np.isfinite(M):
        return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    xs = x[mask]
    ys = y[mask]

    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]

    y_min = np.nanmin(ys)
    y_max = np.nanmax(ys)
    if M < y_min or M > y_max:
        return np.nan

    return float(np.interp(float(M), ys, xs))


def interp_y_at_x(xq: float, x: np.ndarray, y: np.ndarray) -> float:
    """Return moment y at rotation x=xq using linear interpolation; NaN if out of range."""
    if not np.isfinite(xq):
        return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    xs = x[mask]
    ys = y[mask]

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    if xq < xs[0] or xq > xs[-1]:
        return np.nan

    return float(np.interp(float(xq), xs, ys))


def build_dataset_from_components(mat: dict) -> np.ndarray:
    """
    Build dataset (N, L, 6) from MR_conn/MR_frame/NormR_frame stored in .mat.
    """
    required = ["MR_conn", "MR_frame", "NormR_frame"]
    missing = [k for k in required if k not in mat]
    if missing:
        raise KeyError(f"Missing keys to build dataset: {missing}. Provide 'dataset' or these components.")

    MR_conn = mat["MR_conn"]
    MR_frame = mat["MR_frame"]
    NormR_frame = mat["NormR_frame"]

    # The variables are typically 1x2 cell: [0]=x, [1]=y
    MR_conn_x = cell_to_matrix(MR_conn[0])
    MR_conn_y = cell_to_matrix(MR_conn[1])
    MR_frame_x = cell_to_matrix(MR_frame[0])
    MR_frame_y = cell_to_matrix(MR_frame[1])
    Norm_x = cell_to_matrix(NormR_frame[0])
    Norm_y = cell_to_matrix(NormR_frame[1])

    Norm_y = np.nan_to_num(Norm_y, nan=1.0)

    def to_NL(a: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"Curve matrix must be 2D, got {a.shape}")
        # Common export: (L,N) where L~136
        if a.shape[0] <= a.shape[1]:
            return a.T
        return a

    Xc = to_NL(MR_conn_x)
    Yc = to_NL(MR_conn_y)
    Xf = to_NL(MR_frame_x)
    Yf = to_NL(MR_frame_y)
    Xn = to_NL(Norm_x)
    Yn = to_NL(Norm_y)

    N, L = Xc.shape
    for name, A in [("Yc", Yc), ("Xf", Xf), ("Yf", Yf), ("Xn", Xn), ("Yn", Yn)]:
        if A.shape != (N, L):
            raise ValueError(f"Shape mismatch: {name} is {A.shape}, expected {(N, L)}")

    dataset = np.stack([Xc, Yc, Xf, Yf, Xn, Yn], axis=2).astype(np.float32)  # (N,L,6)
    dataset = np.nan_to_num(dataset, nan=1.0)
    return dataset


def build_MR_boltcon(dataset: np.ndarray, Mp_raw: np.ndarray):
    """
    Build MR_boltcon from dataset and Mp.

    dataset channels:
      0: MR_conn.x
      1: MR_conn.y (normalized)
      2: MR_frame.x
      3: MR_frame.y (unused)
      4: NormR_frame.x (unused)
      5: NormR_frame.y (unused)
    """
    dataset = np.asarray(dataset, dtype=np.float32)
    if dataset.ndim != 3 or dataset.shape[2] < 3:
        raise ValueError(f"dataset must be (N, L, >=3). Got {dataset.shape}")

    x_conn = dataset[:, :, 0]
    x_frame = dataset[:, :, 2]
    y_conn_norm = dataset[:, :, 1]

    N = dataset.shape[0]
    Mp_raw = np.squeeze(np.array(Mp_raw)).astype(float)
    if Mp_raw.ndim == 2:
        Mp_raw = Mp_raw.reshape(-1)
    if Mp_raw.size != N:
        raise ValueError(f"Mp size mismatch: {Mp_raw.size} vs N={N}")

    y_conn = y_conn_norm * Mp_raw.reshape(-1, 1)
    x_bolt = x_conn + x_frame
    y_bolt = y_conn

    MR_boltcon = np.stack([x_bolt, y_bolt], axis=2)  # (N,L,2)

    Mp = np.empty(N, dtype=np.float32)
    for i in range(N):
        yi = y_bolt[i]
        yi = yi[np.isfinite(yi)]
        Mp[i] = float(yi.max()) if yi.size else np.nan

    return MR_boltcon, x_bolt, y_bolt, Mp


def run_Module1(
    mat_path: str,
    system: str,
    applied_moment: float,
    frame_length: float,
    comb: int,
    *,
    condition: int | None = None,
    t_IMF: float = 0.02,
    t_SMF: float = 0.04,
    length_col: int | None = None,
    length_tol: float = 0.0,
    progress_bar=None,
    status_text=None,
):
    """
    Run Module 1 and return (DataFrame, summary dict).
    """
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Infer condition if not given
    if condition is None:
        condition = 2 if "SMF" in str(system).upper() else 1

    if progress_bar is not None:
        progress_bar.progress(2)
    if status_text is not None:
        status_text.text("Loading arrays from .mat ...")

    # --- var ---
    if "var" not in mat:
        raise KeyError("The .mat file must contain 'var' (design variable matrix).")
    var = np.array(mat["var"], dtype=np.float32)
    if var.ndim != 2:
        raise ValueError(f"'var' must be 2D. Got {var.shape}")
    if var.shape[0] < var.shape[1]:
        var = var.T
    N = var.shape[0]

    # --- Mp ---
    if "Mp" not in mat:
        raise KeyError("The .mat file must contain 'Mp' (plastic moment).")
    Mp_raw = np.array(mat["Mp"], dtype=np.float32)
    Mp_raw = np.squeeze(Mp_raw)
    if Mp_raw.ndim == 2:
        Mp_raw = Mp_raw.reshape(-1)
    if Mp_raw.size != N:
        raise ValueError(f"Mp size mismatch: {Mp_raw.size} vs N={N}")

    # --- dataset ---
    if "dataset" in mat:
        dataset = np.array(mat["dataset"], dtype=np.float32)
        if dataset.ndim != 3:
            raise ValueError(f"'dataset' must be 3D. Got {dataset.shape}")
        # If exported as (L,N,6)
        if dataset.shape[0] != N and dataset.shape[1] == N:
            dataset = np.transpose(dataset, (1, 0, 2))
    else:
        dataset = build_dataset_from_components(mat)

    if dataset.shape[0] != N:
        raise ValueError(f"N mismatch: dataset N={dataset.shape[0]} but var N={N}")

    if progress_bar is not None:
        progress_bar.progress(10)
    if status_text is not None:
        status_text.text("Building M–R backbone and interpolating ...")

    MR_boltcon, x_bolt, y_bolt, Mp = build_MR_boltcon(dataset, Mp_raw=Mp_raw)
    M_req = 0.8 * Mp
    M_applied = float(applied_moment)

    # Interpolate
    M_IMF = np.empty(N, dtype=np.float32)
    M_SMF = np.empty(N, dtype=np.float32)
    x_at_M = np.empty(N, dtype=np.float32)

    for i in range(N):
        M_IMF[i] = interp_y_at_x(t_IMF, x_bolt[i], y_bolt[i])
        M_SMF[i] = interp_y_at_x(t_SMF, x_bolt[i], y_bolt[i])
        x_at_M[i] = interp_x_at_y(M_applied, x_bolt[i], y_bolt[i])

        if progress_bar is not None and (i % max(1, N // 100) == 0):
            p = 10 + int(55 * (i / max(N - 1, 1)))  # 10~65
            progress_bar.progress(min(p, 65))
            if status_text is not None:
                status_text.text(f"Interpolating: {i:,} / {N:,}")

    if progress_bar is not None:
        progress_bar.progress(70)
    if status_text is not None:
        status_text.text("Filtering candidates ...")

    # Frame-length filter
    if length_col is not None:
        L_vec = var[:, int(length_col)].astype(float)
        L_med = np.nanmedian(L_vec[np.isfinite(L_vec)]) if np.isfinite(L_vec).any() else np.nan
        L = float(frame_length)
        tol = float(length_tol)

        # Auto-scale (m -> mm) if needed
        if np.isfinite(L_med) and L_med > 100 and L < 10:
            L *= 1000.0
            tol *= 1000.0

        if tol > 0:
            ok_len = np.isfinite(L_vec) & (np.abs(L_vec - L) <= tol)
        else:
            ok_len = np.isfinite(L_vec) & (L_vec == L)
    else:
        ok_len = np.ones(N, dtype=bool)

    base = np.isfinite(M_req) & np.isfinite(x_at_M) & ok_len & (M_applied >= M_req)

    if int(condition) == 1:
        cond = base & np.isfinite(M_IMF) & (M_IMF >= M_req)
    elif int(condition) == 2:
        cond = base & np.isfinite(M_IMF) & (M_IMF >= M_req) & np.isfinite(M_SMF) & (M_SMF >= M_req)
    else:
        raise ValueError("condition must be 1 (IMF) or 2 (SMF).")

    idx_ok = np.where(cond)[0]
    if idx_ok.size == 0:
        return pd.DataFrame(), {"N_total": int(N), "N_ok": 0, "comb_returned": 0}

    if progress_bar is not None:
        progress_bar.progress(80)
    if status_text is not None:
        status_text.text("Scoring and selecting Top-N ...")

    # Score (as your original)
    A = (var[:, 0] * var[:, 1]) - (
        (var[:, 1] - 2 * var[:, 3]) * (var[:, 0] - 2 * var[:, 3])
        - (var[:, 0] - 2 * var[:, 2]) * var[:, 3]
    )
    V = A * var[:, 10]
    nb = var[:, 7] + var[:, 8]
    mr = (M_applied - M_req) / M_req

    score = nb[idx_ok] + V[idx_ok] + mr[idx_ok]
    idx_sort = idx_ok[np.argsort(score)]
    idx_top = idx_sort[: min(int(comb), idx_sort.size)]

    if progress_bar is not None:
        progress_bar.progress(92)
    if status_text is not None:
        status_text.text("Preparing output table ...")

    df_meta = pd.DataFrame({"M_req(0.8Mp)": M_req[idx_top],"margin": mr[idx_top],"nb": nb[idx_top],"Volume": V[idx_top],
            "score(nb+V+margin)": (nb[idx_top] + V[idx_top] + (1-mr[idx_top])),"M_IMF(0.02rad)": M_IMF[idx_top],"M_SMF(0.04rad)": M_SMF[idx_top]})

    cols_var = None

    if "index_input" in mat:
        try:
            names = MatList(mat["index_input"])
            # Use only if the length matches var columns
            if len(names) == var.shape[1]:
                cols_var = names
        except Exception:
            cols_var = None

    # Fallback
    if cols_var is None:
        cols_var = [f"var_{j}" for j in range(var.shape[1])]

    df_var = pd.DataFrame(var[idx_top, :], columns=cols_var)
    out = pd.concat([df_meta, df_var], axis=1)

    summary = {
        "N_total": int(N),
        "N_ok": int(idx_ok.size),
        "comb_returned": int(out.shape[0]),
    }

    if progress_bar is not None:
        progress_bar.progress(100)
    if status_text is not None:
        status_text.text("Done.")

    return out, summary
