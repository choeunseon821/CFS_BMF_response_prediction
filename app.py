from __future__ import annotations

from pathlib import Path
import io
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

from modules.screening import run_Module1
from modules.common import resolve_database, load_design_table, get_domain, validate_domain, rank_candidates
from modules.ml_crp import load_bundle as load_ml_bundle, predict as predict_ml, postprocess as postprocess_ml, train_and_save
from modules.dl_full import INPUT_NAMES, load_assets as load_dl_assets, predict_array, to_dataframe, moment_at_rotation
from modules.plots import plot_dl_response, fig_png_bytes
from modules.report import build_pdf_report

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
ASSET_DIR = ROOT / "assets"

LANDING_LOGO = ASSET_DIR / "logo2.png"
FAVICON_LOGO = ASSET_DIR / "AXIS logo.png"

ML_BUNDLE = MODEL_DIR / "crp_models.joblib"
DL_MODEL = MODEL_DIR / "BCRP_script.pt"
DL_X = MODEL_DIR / "BCRP_dataset_X.pkl"
DL_Y = MODEL_DIR / "BCRP_dataset_Y.pkl"

if FAVICON_LOGO.is_file():
    favicon = Image.open(FAVICON_LOGO)
    st.set_page_config(page_title="CFS-BMF Design Platform",page_icon=favicon,layout="wide",initial_sidebar_state="collapsed")
else:
    st.set_page_config(page_title="CFS-BMF Design Platform",layout="wide",initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# Presentation
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] {display:none !important;}
      .block-container {max-width: 1450px; padding-top: 2.0rem; padding-bottom: 4rem;}
      .axis-title {font-size: 2.4rem; font-weight: 750; letter-spacing: -0.02em; margin-bottom: .25rem;}
      .axis-sub {color: #7a7f87; font-size: .95rem; margin-bottom: 1.5rem;}
      .step-card {border:1px solid #e6e8eb; border-radius:14px; padding:1.1rem 1.2rem; margin:.65rem 0 1rem 0; background:#fff;}
      .step-lock {color:#8b9097; font-size:.92rem;}
      .home-wrap {text-align:center; padding-top: 2vh;}
      div.stButton > button[kind="primary"] {border-radius:9px; min-height: 44px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    db = resolve_database(DATA_DIR)
    defaults = {
        "entered": False,
        "db_path": str(db) if db else None,
        "ranked": None,
        "selected_design": None,
        "ml_result": None,
        "dl_result": None,
        "dl_fig_png": None,
        "system": "Special moment frame, SMF",
        "applied_moment": 50.0,
        "frame_length": 2.0,
        "top_n": 20,
        "ranking_mode": "Balanced",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()


@st.cache_data(show_spinner=False)
def cached_design_table(path: str):
    return load_design_table(path)


@st.cache_resource(show_spinner=False)
def cached_ml_bundle(path: str):
    return load_ml_bundle(path)


@st.cache_resource(show_spinner=False)
def cached_dl_assets(model: str, sx: str, sy: str):
    return load_dl_assets(model, sx, sy)


@st.cache_resource(show_spinner=False)
def ensure_ml_model(db_path: str, out_path: str):
    out = Path(out_path)
    if out.is_file():
        return True, None
    try:
        train_and_save(Path(db_path), out)
        return out.is_file(), None
    except Exception as exc:
        return False, str(exc)


def current_db() -> Path | None:
    p = st.session_state.get("db_path")
    return Path(p) if p and Path(p).is_file() else None


def selected_row_df() -> pd.DataFrame | None:
    d = st.session_state.get("selected_design")
    return pd.DataFrame([d]) if d else None


def reset_after_screening():
    st.session_state["selected_design"] = None
    st.session_state["ml_result"] = None
    st.session_state["dl_result"] = None
    st.session_state["dl_fig_png"] = None


def reset_after_candidate():
    st.session_state["ml_result"] = None
    st.session_state["dl_result"] = None
    st.session_state["dl_fig_png"] = None


def dl_ready() -> bool:
    return all(p.is_file() for p in (DL_MODEL, DL_X, DL_Y))


def save_uploaded_model(uploaded) -> bool:
    if uploaded is None:
        return False
    MODEL_DIR.mkdir(exist_ok=True)
    DL_MODEL.write_bytes(uploaded.getbuffer())
    cached_dl_assets.clear()
    return DL_MODEL.is_file()


def top_header():
    st.markdown('<div class="axis-title">CFS-BMF Design & Response Prediction Platform</div>',unsafe_allow_html=True)
    st.markdown('<div class="axis-sub">Database screening · candidate editing · ML critical response points · DL full response · report export</div>',unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# LANDING / HOME
# -----------------------------------------------------------------------------
if not st.session_state["entered"]:
    st.markdown('<div class="home-wrap">', unsafe_allow_html=True)

    if LANDING_LOGO.is_file():
        a, b, c = st.columns([1.2, 1, 1.2])
        with b:
            st.image(str(LANDING_LOGO), use_container_width=True)

    st.markdown("## CFS-BMF Design & Response Prediction Platform")
    st.caption("Integrated screening, ML critical-response prediction, ""DL full-response prediction, and design reporting")

    db = current_db()

    if db is None:
        st.warning("Place your database .mat file in the /data folder ""before entering the platform.")

    st.write("")

    if st.button("Enter Platform",type="primary",use_container_width=True,disabled=(db is None)):
        st.session_state["entered"] = True
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

top_header()

# Prepare ML automatically only after entering the platform.
db = current_db()
ml_auto_error = None
if db is not None and not ML_BUNDLE.is_file():
    with st.spinner("First run: preparing the ML critical-response model from the database. This is saved and reused afterwards..."):
        _, ml_auto_error = ensure_ml_model(str(db), str(ML_BUNDLE))

# -----------------------------------------------------------------------------
# STEP 1 — SCREENING
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("1. Design Screening")
if db is None:
    st.error("Database is unavailable. Put the .mat database in /data and restart.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    system = st.selectbox("Moment-frame system", ["Intermediate moment frame, IMF", "Special moment frame, SMF"], index=1 if "SMF" in st.session_state["system"] else 0)
with c2:
    applied_moment = st.number_input("Applied moment, M", min_value=0.0, value=float(st.session_state["applied_moment"]), step=1.0)
with c3:
    frame_opts = [2.0, 2.5, 3.0]
    old_frame = float(st.session_state["frame_length"])
    frame_length = st.selectbox("Frame length [m]", frame_opts, index=frame_opts.index(old_frame) if old_frame in frame_opts else 0)
with c4:
    top_opts = [10, 20, 30, 50, 100]
    old_top = int(st.session_state["top_n"])
    top_n = st.selectbox("Number of recommendations", top_opts, index=top_opts.index(old_top) if old_top in top_opts else 1)

ranking_modes = ["Balanced", "Performance", "Material efficiency"]
ranking = st.radio("Ranking basis", ranking_modes, horizontal=True, index=ranking_modes.index(st.session_state["ranking_mode"]))
st.info("Feasibility filtering follows the Module 1 logic: M ≥ 0.80Mp and the IMF/SMF drift requirement. The feasible set is then sorted by the selected ranking basis.")

if st.button("Run Screening", type="primary", use_container_width=True):
    with st.spinner("Screening database candidates..."):
        try:
            raw, summary = run_Module1(
                mat_path=str(db), system=system, applied_moment=float(applied_moment),
                frame_length=float(frame_length), comb=1_000_000, length_col=10, length_tol=0.0,
            )
            ranked = rank_candidates(raw, ranking, int(top_n))
            st.session_state.update({
                "ranked": ranked, "system": system, "applied_moment": float(applied_moment),
                "frame_length": float(frame_length), "top_n": int(top_n), "ranking_mode": ranking,
            })
            reset_after_screening()
            st.success(f"{summary.get('N_ok', len(raw)):,} feasible combinations found; Top {len(ranked)} are shown below.")
        except Exception as exc:
            st.error(f"Screening failed: {exc}")

ranked = st.session_state.get("ranked")
if isinstance(ranked, pd.DataFrame) and not ranked.empty:
    st.subheader("Recommended combinations")
    show_cols = [c for c in ["Rank", "ranking_score", "M_req(0.8Mp)", "M_IMF(0.02rad)", "M_SMF(0.04rad)", "nb", "Volume", "margin"] if c in ranked.columns]
    show_cols += [c for c in ranked.columns if c not in show_cols][:14]
    st.dataframe(ranked[show_cols], use_container_width=True, hide_index=True, height=390)
    st.download_button("Download ranked candidates (CSV)", ranked.to_csv(index=False).encode("utf-8-sig"), "screening_candidates.csv", "text/csv")

# -----------------------------------------------------------------------------
# STEP 2 — CANDIDATE EXPLORER
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("2. Candidate Selection & Editing")
if not isinstance(ranked, pd.DataFrame) or ranked.empty:
    st.caption("Complete Step 1 to unlock candidate selection.")
else:
    db_df = cached_design_table(str(db))
    domain = get_domain(db_df)
    rank_choice = st.selectbox("Candidate rank", ranked["Rank"].tolist(), key="candidate_rank")
    row = ranked.loc[ranked["Rank"] == rank_choice].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Ranking score", f"{float(row['ranking_score']):.4f}" if "ranking_score" in row else "-")
    m2.metric("Member volume index", f"{float(row['Volume']):.4g}" if "Volume" in row else "-")
    m3.metric("Bolt count", f"{float(row['nb']):.0f}" if "nb" in row else "-")

    meta_cols = {"Rank", "ranking_score", "M_req(0.8Mp)", "margin", "nb", "Volume", "score(nb+V+margin)", "M_IMF(0.02rad)", "M_SMF(0.04rad)"}
    design_cols = [c for c in db_df.columns if c in ranked.columns and c not in meta_cols]
    edited = {}
    with st.expander("Edit design variables within the database domain", expanded=True):
        for i in range(0, len(design_cols), 4):
            cols = st.columns(4)
            for j, name in enumerate(design_cols[i:i+4]):
                drow = domain.loc[domain["Variable"] == name]
                lo = float(drow["Min"].iloc[0]) if not drow.empty else None
                hi = float(drow["Max"].iloc[0]) if not drow.empty else None
                val = float(pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0])
                with cols[j]:
                    edited[name] = st.number_input(name, value=val, format="%.6g", key=f"edit_{rank_choice}_{name}")
                    if lo is not None and hi is not None:
                        st.caption(f"DB: {lo:g} – {hi:g}")

    valid, errors = validate_domain(edited, domain)
    if not valid:
        st.error("Outside database/model domain: " + " | ".join(errors[:8]))

    chosen = row.to_dict(); chosen.update(edited)
    if st.button("Confirm Design & Continue", type="primary", disabled=not valid, use_container_width=True):
        st.session_state["selected_design"] = chosen
        reset_after_candidate()
        st.success("Design confirmed. ML prediction is now available below.")
        st.rerun()

# -----------------------------------------------------------------------------
# STEP 3 — ML
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("3. ML Critical Response Point Prediction")
design_df = selected_row_df()
if design_df is None:
    st.caption("Confirm a design in Step 2 to unlock ML prediction.")
elif not ML_BUNDLE.is_file():
    st.error("ML model preparation failed." + (f" Details: {ml_auto_error}" if ml_auto_error else ""))
else:
    if st.button("Run ML Prediction", type="primary", use_container_width=True):
        try:
            bundle = cached_ml_bundle(str(ML_BUNDLE))
            st.session_state["ml_result"] = postprocess_ml(predict_ml(design_df, bundle))
            st.success("ML critical-response prediction completed.")
        except Exception as exc:
            st.error(f"ML prediction failed: {exc}")

    ml = st.session_state.get("ml_result")
    if isinstance(ml, pd.DataFrame) and not ml.empty:
        ycols = [c for c in ml.columns if c not in design_df.columns]
        st.dataframe(ml[ycols].T.rename(columns={0: "Predicted value"}), use_container_width=True)
        pairs = [
            ("Connection", [("Rc_slip", "Mc_slip"), ("Rc_peak", "Mc_peak")]),
            ("Frame", [("Rf_yielding", "Mf_yielding"), ("Rf_peak", "Mf_peak"), ("Rf_softening", "Mf_softening")]),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (title, pp) in zip(axes, pairs):
            pts = [(0.0, 0.0)]
            for xn, yn in pp:
                if xn in ml.columns and yn in ml.columns:
                    x = float(ml.iloc[0][xn]); y = float(ml.iloc[0][yn])
                    if np.isfinite(x) and np.isfinite(y): pts.append((x, y))
            if len(pts) > 1:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
            ax.set_title(title); ax.set_xlabel("Rotation [rad]"); ax.set_ylabel("Normalized moment"); ax.grid(True, alpha=.25)
        fig.tight_layout(); st.pyplot(fig)
        st.download_button("Download ML CRPs (CSV)", ml.to_csv(index=False).encode("utf-8-sig"), "ml_critical_response_points.csv", "text/csv")

# -----------------------------------------------------------------------------
# STEP 4 — DL
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("4. DL Full Response Prediction")
ml = st.session_state.get("ml_result")
if design_df is None:
    st.caption("Confirm a design in Step 2 first.")
elif not isinstance(ml, pd.DataFrame):
    st.caption("Complete Step 3 to unlock DL prediction.")
else:
    if not DL_MODEL.is_file():
        st.warning("DL model was not found automatically. Select BCRP_script.pt once below; the app saves it locally and reuses it on later runs.")
        up_model = st.file_uploader("Select BCRP_script.pt", type=["pt"], key="inline_dl_upload")
        if up_model is not None and st.button("Save DL Model"):
            if save_uploaded_model(up_model):
                st.success("DL model saved successfully.")
                st.rerun()
    elif not DL_X.is_file() or not DL_Y.is_file():
        st.error("The scaler bundle is incomplete. Re-extract the platform ZIP because both scaler files are included in the package.")
    else:
        db_df = cached_design_table(str(db))
        missing_inputs = [c for c in INPUT_NAMES if c not in design_df.columns]
        if missing_inputs:
            st.error("Selected design is missing DL inputs: " + ", ".join(missing_inputs))
        else:
            dl_domain = get_domain(db_df, INPUT_NAMES)
            vals = {c: float(design_df.iloc[0][c]) for c in INPUT_NAMES}
            valid, errors = validate_domain(vals, dl_domain)
            if not valid:
                st.error("DL prediction is disabled outside the database domain: " + " | ".join(errors[:8]))
            elif st.button("Run DL Full-Response Prediction", type="primary", use_container_width=True):
                try:
                    model, sx, sy, device = cached_dl_assets(str(DL_MODEL), str(DL_X), str(DL_Y))
                    X = np.array([[vals[c] for c in INPUT_NAMES]], dtype=np.float32)
                    pred = predict_array(X, model, sx, sy, device)[0]
                    st.session_state["dl_result"] = pred
                    st.session_state["dl_fig_png"] = fig_png_bytes(plot_dl_response(pred, st.session_state["system"]))
                    st.success("DL full-response prediction completed.")
                except Exception as exc:
                    st.error(f"DL prediction failed: {exc}")

        pred = st.session_state.get("dl_result")
        if isinstance(pred, np.ndarray):
            st.pyplot(plot_dl_response(pred, st.session_state["system"]), use_container_width=True)
            m04 = moment_at_rotation(pred, 0.04)
            drift = 0.04 if "SMF" in st.session_state["system"].upper() else 0.02
            mdrift = moment_at_rotation(pred, drift)
            a, b = st.columns(2)
            a.metric("Normalized moment at 0.04 rad", "N/A" if not np.isfinite(m04) else f"{m04:.3f}")
            b.metric(f"Normalized moment at {drift:.2f} rad", "N/A" if not np.isfinite(mdrift) else f"{mdrift:.3f}")
            pred_df = to_dataframe(pred)
            with st.expander("View full predicted data"):
                st.dataframe(pred_df, use_container_width=True, height=380)
            st.download_button("Download full DL response (CSV)", pred_df.to_csv(index=False).encode("utf-8-sig"), "dl_full_response.csv", "text/csv")
            if st.session_state.get("dl_fig_png"):
                st.download_button("Download response plot (PNG)", st.session_state["dl_fig_png"], "dl_response.png", "image/png")

# -----------------------------------------------------------------------------
# STEP 5 — REPORT
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("5. Design Report")
pred = st.session_state.get("dl_result")
if not isinstance(pred, np.ndarray):
    st.caption("Complete Step 4 to generate the final report.")
else:
    system = st.session_state["system"]
    m04 = moment_at_rotation(pred, 0.04)
    drift = 0.04 if "SMF" in system.upper() else 0.02
    mdrift = moment_at_rotation(pred, drift)
    summary = {
        "Structural system": system,
        "Applied moment": st.session_state["applied_moment"],
        "Frame length [m]": st.session_state["frame_length"],
        "Ranking basis": st.session_state["ranking_mode"],
        "DL M/Mp at 0.04 rad": "N/A" if not np.isfinite(m04) else f"{m04:.4f}",
        f"DL M/Mp at {drift:.2f} rad": "N/A" if not np.isfinite(mdrift) else f"{mdrift:.4f}",
        "ML CRPs": "Available",
        "DL full response": "Available",
        "Failure mode": "Classification rule not yet connected",
    }
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("System", "SMF" if "SMF" in system.upper() else "IMF")
    c2.metric("Applied M", f"{st.session_state['applied_moment']:.2f}")
    c3.metric("M/Mp @ 0.04 rad", "-" if not np.isfinite(m04) else f"{m04:.3f}")
    c4.metric("Prediction domain", "Within DB range")
    st.dataframe(pd.DataFrame(summary.items(), columns=["Item", "Result"]), hide_index=True, use_container_width=True)
    st.pyplot(plot_dl_response(pred, system), use_container_width=True)
    report_inputs = [(c, design_df.iloc[0][c]) for c in INPUT_NAMES if c in design_df.columns]
    pdf = build_pdf_report(summary, report_inputs, st.session_state.get("dl_fig_png"))
    st.download_button("Download PDF Report", pdf, "CFS_BMF_design_report.pdf", "application/pdf", type="primary", use_container_width=True)

# Compact diagnostic footer—not a workflow page.
st.markdown("---")
with st.expander("System status / maintenance"):
    st.write({
        "Database": str(db) if db else "Missing",
        "ML model": "Ready" if ML_BUNDLE.is_file() else "Missing",
        "DL model": "Ready" if DL_MODEL.is_file() else "Missing",
        "DL X scaler": "Ready" if DL_X.is_file() else "Missing",
        "DL Y scaler": "Ready" if DL_Y.is_file() else "Missing",
    })
    if st.button("Return to Home"):
        st.session_state["entered"] = False
        st.rerun()
