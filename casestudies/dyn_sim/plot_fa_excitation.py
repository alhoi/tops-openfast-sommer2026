"""Tower fore-aft (FA) grid-excitation curve, support off and on.

Reads full_matrix/fa/ (fast.fmu, fmu_YawBrTAxp) and plots the settled
tower-top FA acceleration at the process-load drive frequency, for both
support off (passive grid-forming floor) and support on. This is a
standalone figure; the SS figure lives in plot_ss_fa_excitation.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP = PROJECT_ROOT / "results" / "em_interaction_sweep" / "full_matrix"
OUT = SWEEP / "fa_excitation.png"
FA_HZ = 0.233   # tower fore-aft mode (near the SS mode; ring-down 0.229)


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


def main() -> None:
    fa_dir = SWEEP / "fa"
    fa_on = curve(fa_dir, "fa", "fmu_YawBrTAxp", "on")
    fa_off = curve(fa_dir, "fa", "fmu_YawBrTAxp", "off")

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14,
        "axes.labelsize": 12.5, "legend.fontsize": 11,
    })
    c_on, c_off = "#2ca02c", "#8c8c8c"

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x_on, y_on = _xy(fa_on)
    x_off, y_off = _xy(fa_off)
    aligned = x_on.size > 0 and np.array_equal(x_on, x_off)

    ax.plot(x_off, y_off, marker="s", ms=6, ls="--", color=c_off, lw=1.6,
            mfc="white", mec=c_off, zorder=3, label="FA, support off")
    ax.plot(x_on, y_on, marker="o", ms=6.5, color=c_on, lw=2.0,
            mec="white", mew=0.8, zorder=4, label="FA, support on")

    # Ring the resonance peak and note the amplification vs support off.
    if x_on.size:
        i = int(np.argmax(y_on))
        fp, yp = x_on[i], y_on[i]
        yo = y_off[i] if aligned else np.nan
        ax.scatter([fp], [yp], s=120, facecolor="none", edgecolor=c_on,
                   lw=1.8, zorder=5)
        txt = f"peak {yp:.3f} m/s$^2$"
        if np.isfinite(yo) and yo > 0:
            txt += f"\n{yp / yo:.1f}$\\times$ support off"
        ax.annotate(txt, xy=(fp, yp), xytext=(0.06, 0.9),
                    textcoords="axes fraction", fontsize=10.5,
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))

    ymax = float(np.nanmax(np.concatenate([y_on, y_off]))) if x_on.size else 1.0
    ax.set_ylim(0.0, ymax * 1.35)
    ax.margins(x=0.03)

    ax.axvline(FA_HZ, color="0.45", ls=":", lw=1.0)
    ax.text(FA_HZ, ax.get_ylim()[1], f" FA mode {FA_HZ} Hz",
            fontsize=9.5, va="top", ha="left", color="0.3")

    ax.set_xlabel("Process-load frequency  [Hz]")
    ax.set_ylabel("Settled tower-top acceleration\nat drive frequency  [m/s$^2$]")
    ax.set_title("Grid excitation of the tower fore-aft mode (OpenFAST, Region 3)")
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved {OUT}")
    print(f"  FA on: {len(fa_on)} pts, FA off: {len(fa_off)} pts"
          + ("" if (fa_on or fa_off) else "  (fa/ sweep not run yet)"))


if __name__ == "__main__":
    main()
