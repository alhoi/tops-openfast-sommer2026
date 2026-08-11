"""Combined tower-mode grid-excitation curves: SS and FA, support off and on.

Overlays the side-to-side (SS, fmu_YawBrTAyp) and fore-aft (FA, fmu_YawBrTAxp)
settled tower-top accelerations from the full_matrix sweep in one figure.
Colour encodes the mode (SS blue, FA green); line style encodes support
(solid = on, dashed = off). The standalone SS and FA figures live in
plot_ss_fa_excitation.py and plot_fa_excitation.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP = PROJECT_ROOT / "results" / "em_interaction_sweep" / "full_matrix"
OUT = SWEEP / "ss_fa_combined.png"
MODE_HZ = 0.233   # first tower bending mode (SS and FA nearly coincide)

C_SS, C_FA = "#1f77b4", "#ff7f0e"


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


def _xy(pts):
    return (np.array([p[0] for p in pts], float),
            np.array([p[1] for p in pts], float))


def _lab(base, pts):
    if pts:
        return f"{base}  ({max(p[1] for p in pts):.3f})"
    return base


def _lab_on(base, on, off):
    if not on:
        return base
    peak = max(p[1] for p in on)
    txt = f"{peak:.3f}"
    if off:
        r = peak / max(p[1] for p in off)
        txt += f", {r:.0f}$\\times$ off" if r >= 10 else f", {r:.1f}$\\times$ off"
    return f"{base}  ({txt})"


def main() -> None:
    ss_dir, fa_dir = SWEEP / "ss", SWEEP / "fa"
    ss_on = curve(ss_dir, "ss", "fmu_YawBrTAyp", "on")
    ss_off = curve(ss_dir, "ss", "fmu_YawBrTAyp", "off")
    fa_on = curve(fa_dir, "fa", "fmu_YawBrTAxp", "on")
    fa_off = curve(fa_dir, "fa", "fmu_YawBrTAxp", "off")

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14,
        "axes.labelsize": 12.5, "legend.fontsize": 10.5,
    })
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax2 = ax.twinx()   # separate right axis for the (much smaller) FA response

    # SS -> left axis (blue)
    l1, = ax.plot(*_xy(ss_on), marker="o", ms=6.5, color=C_SS, lw=2.0,
                  mec="white", mew=0.8, zorder=5,
                  label=_lab_on("SS, support on", ss_on, ss_off))
    l2, = ax.plot(*_xy(ss_off), marker="o", ms=6, ls="--", color=C_SS, lw=1.6,
                  mfc="white", mec=C_SS, zorder=4,
                  label=_lab("SS, support off", ss_off))
    # FA -> right axis (orange)
    l3, = ax2.plot(*_xy(fa_on), marker="s", ms=6.5, color=C_FA, lw=2.0,
                   mec="white", mew=0.8, zorder=5,
                   label=_lab_on("FA, support on", fa_on, fa_off))
    l4, = ax2.plot(*_xy(fa_off), marker="s", ms=6, ls="--", color=C_FA, lw=1.6,
                   mfc="white", mec=C_FA, zorder=4,
                   label=_lab("FA, support off", fa_off))

    ss_y = [p[1] for p in (ss_on + ss_off)]
    fa_y = [p[1] for p in (fa_on + fa_off)]
    ax.set_ylim(0.0, (float(np.nanmax(ss_y)) if ss_y else 1.0) * 1.3)
    ax2.set_ylim(0.0, (float(np.nanmax(fa_y)) if fa_y else 1.0) * 1.3)
    ax.margins(x=0.03)

    ax.axvline(MODE_HZ, color="0.45", ls=":", lw=1.0, zorder=1)
    if ss_on:
        j = int(np.argmax([p[1] for p in ss_on]))
        fp_ss, yp_ss = ss_on[j]
    else:
        fp_ss, yp_ss = MODE_HZ, ax.get_ylim()[1] * 0.9
    ax.annotate(f"tower mode {MODE_HZ} Hz", xy=(fp_ss, yp_ss),
                xytext=(-10, 11), textcoords="offset points",
                ha="right", va="center", fontsize=9.5, color="0.3")

    ax.set_xlabel("Process-load frequency  [Hz]")
    ax.set_ylabel("SS tower-top acceleration  [m/s$^2$]", color=C_SS)
    ax2.set_ylabel("FA tower-top acceleration  [m/s$^2$]", color=C_FA)
    ax.tick_params(axis="y", colors=C_SS)
    ax2.tick_params(axis="y", colors=C_FA)
    ax.spines["left"].set_color(C_SS)
    ax2.spines["right"].set_color(C_FA)
    ax2.spines["left"].set_visible(False)
    ax.set_title("Grid excitation of the tower modes (OpenFAST, Region 3)")
    ax.grid(True, alpha=0.25, lw=0.7)

    lines = [l1, l2, l3, l4]
    ax.legend(lines, [ln.get_label() for ln in lines],
              frameon=True, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}")
    print(f"  SS on/off: {len(ss_on)}/{len(ss_off)} pts, "
          f"FA on/off: {len(fa_on)}/{len(fa_off)} pts")


if __name__ == "__main__":
    main()
