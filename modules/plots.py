from __future__ import annotations
import io
import numpy as np
import matplotlib.pyplot as plt


def plot_dl_response(prediction: np.ndarray, system: str):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    titles = ["Connection", "Frame", "Normalized rotation", "Total"]
    pairs = [(0,1), (2,3), (4,5)]
    for i, (xidx, yidx) in enumerate(pairs):
        axes[i].plot(prediction[:, xidx], prediction[:, yidx], linewidth=1.5)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Rotation [rad]")
        axes[i].set_ylabel("Normalized response")
        axes[i].grid(True, alpha=0.25)
        axes[i].set_xlim(0, 0.1)
    axes[3].plot(prediction[:, 0] + prediction[:, 2], prediction[:, 3], linewidth=1.5)
    axes[3].set_title(titles[3])
    axes[3].set_xlabel("Rotation [rad]")
    axes[3].set_ylabel("Normalized moment")
    axes[3].grid(True, alpha=0.25)
    axes[3].set_xlim(0, 0.1)
    drift = 0.04 if "SMF" in system.upper() else 0.02
    axes[3].axvline(drift, linestyle="--", linewidth=1.0)
    fig.tight_layout()
    return fig


def fig_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=250, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
