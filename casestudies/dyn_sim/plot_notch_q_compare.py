r"""Compare the support-path notch at Q8 vs Q4 on the GT-trip event.

Droop+inertia (8e7, capped burst) with: no notch / notch Q8 / notch Q4.
Prints the settled-window ripple (30-60 s, detrended: peak-to-peak, RMS,
dominant frequency) and saves a NEW figure (does not overwrite any existing
plot).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
D = PROJECT_ROOT / "results" / "em_interaction_sweep" / "freq_support_3way"
OUT = D / "freq_support_notch_q_compare.png"

# (stem, label, colour)
CASES = [
    ("deload_droop_inertia_cap",   "droop + inertia (no notch)", "#2ca02c"),
    ("deload_droop_inertia_capq8", "+ notch Q8",                 "#9467bd"),
    ("deload_droop_inertia_capq4", "+ notch Q4",                 "#d62728"),
]
LO, HI = 30.0, 60.0   # settled ripple window
EVENT_TIME = 20.0


def ripple(df: pd.DataFrame):
    t = df["t"].to_numpy()
    m = (t >= LO) & (t <= HI)
    tt = t[m]
    P = 100.0 * df["P_e_sys_pu"].to_numpy()[m]
    A = np.vstack([tt, np.ones_like(tt)]).T
    c, _, _, _ = np.linalg.lstsq(A, P, rcond=None)
    Pd = P - A @ c
    pp = Pd.max() - Pd.min()
    rms = np.sqrt(np.mean(Pd**2))
    dt = np.median(np.diff(tt))
    w = np.hanning(len(Pd))
    sp = np.abs(np.fft.rfft((Pd - Pd.mean()) * w))
    fr = np.fft.rfftfreq(len(Pd), dt)
    k = np.argmax(sp[1:]) + 1
    return pp, rms, fr[k]


def main() -> None:
    frames = {}
    for stem, *_ in CASES:
        fp = D / f"{stem}.csv"
        if fp.exists():
            frames[stem] = pd.read_csv(fp)
        else:
            print(f"  (missing {fp.name})")

    print(f"Settled ripple {LO:.0f}-{HI:.0f} s (detrended power):")
    print(f"{'case':>28} {'P_pp[MW]':>9} {'P_rms[MW]':>10} {'f_dom[Hz]':>10}")
    for stem, label, _c in CASES:
        if stem in frames:
            pp, rms, fd = ripple(frames[stem])
            print(f"{label:>28} {pp:>9.4f} {rms:>10.4f} {fd:>10.3f}")

    fig, (ax_f, ax_p) = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
    for stem, label, colour in CASES:
        if stem not in frames:
            continue
        df = frames[stem]
        t = df["t"].to_numpy()
        ax_f.plot(t, df["f_grid_hz"], color=colour, lw=1.5, label=label)
        ax_p.plot(t, 100.0 * df["P_e_sys_pu"], color=colour, lw=1.5, label=label)

    for ax in (ax_f, ax_p):
        ax.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    ax_p.axhline(15.0, color="k", ls="--", lw=0.9)
    ax_p.text(60, 15.0, " 15 MW rating", va="bottom", ha="right", fontsize=8)
    ax_f.set_ylabel("Grid frequency  $f_{grid}$  [Hz]")
    ax_p.set_ylabel("WT electrical power  [MW]")
    ax_p.set_xlabel("Time [s]")
    ax_f.set_title("Support-path notch: Q8 vs Q4 (droop + inertia 8e7, "
                   "capped burst)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
