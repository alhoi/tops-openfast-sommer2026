r"""
Tower-mode excitation figure: does a grid process-load reach the tower
side-to-side (SS) and fore-aft (FA) modes, and only with frequency support?

Reads the OpenFAST full-matrix sweep CSVs and plots the settled tower-top
acceleration at the drive frequency vs the process-load frequency:
  SS (YawBrTAyp) with support on and off  (from full_matrix/ss/),
  FA (YawBrTAxp) with support on           (from full_matrix/fa/, when it exists).

SS resonates sharply only with support on; the fore-aft mode (thrust-driven)
stays flat, showing the grid->tower coupling is controller-mediated and
selective. FA is drawn only once the fa/ sweep has been run.

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_ss_fa_excitation.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP = PROJECT_ROOT / "results" / "em_interaction_sweep" / "full_matrix"
OUT = SWEEP / "ss_fa_excitation.png"
SS_HZ = 0.233   # forced-resonance peak from the refined sweep (ring-down 0.229 agrees within its resolution)


def lockin(t, sig, f, t_lo, t_hi) -> float:
    m = (t >= t_lo) & (t <= t_hi)
    tt, s = t[m], np.asarray(sig[m], float)
    if tt.size < 8:
        return float("nan")
    s = s - s.mean()
    w = 2.0 * np.pi * f
    c = np.trapezoid(s * np.cos(w * tt), tt)
    q = np.trapezoid(s * np.sin(w * tt), tt)
    span = tt[-1] - tt[0]
    return 2.0 * np.hypot(c, q) / span if span > 0 else float("nan")


def curve(folder: Path, prefix: str, signal: str, sup: str):
    pts = []
    for fp in sorted(folder.glob(f"{prefix}_f*_{sup}.csv")):
        m = re.search(r"_f([0-9]+p[0-9]+)_", fp.stem)
        if not m:
            continue
        f = float(m.group(1).replace("p", "."))
        df = pd.read_csv(fp)
        if signal not in df or "t" not in df:
            continue
        t = df["t"].to_numpy()
        a = lockin(t, df[signal].to_numpy(), f, 0.6 * t.max(), t.max())
        pts.append((f, a))
    return sorted(pts)


def plot_curve(ax, pts, **kw):
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)


def main() -> None:
    ss_on = curve(SWEEP / "ss", "ss", "fmu_YawBrTAyp", "on")
    ss_off = curve(SWEEP / "ss", "ss", "fmu_YawBrTAyp", "off")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plot_curve(ax, ss_on, marker="o", color="#1f77b4", lw=1.7,
               label="SS, support on")
    plot_curve(ax, ss_off, marker="o", ls="--", color="#7f7f7f", lw=1.3,
               label="SS, support off")

    ax.axvline(SS_HZ, color="k", ls=":", lw=0.8)
    ax.text(SS_HZ, ax.get_ylim()[1], f"  SS mode {SS_HZ} Hz",
            fontsize=8, va="top", ha="left")
    ax.set_xlabel("Process-load frequency [Hz]")
    ax.set_ylabel("Settled tower-top acceleration\nat drive frequency  [m/s$^2$]")
    ax.set_title("Grid excitation of the tower side-to-side mode (OpenFAST, Region 3)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}  (SS only)")
    print(f"  SS on: {len(ss_on)} pts, SS off: {len(ss_off)} pts")


if __name__ == "__main__":
    main()
