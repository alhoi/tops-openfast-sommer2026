r"""
Mechanical response of the OpenFAST turbine during the de-loaded gas-turbine-trip
frequency-support runs produced by freq_support_3way.py.

Plots the internal FMU mechanics (rotor speed, blade pitch, generator torque and
high-speed-shaft torque) for the three cases (no support, droop, droop + synthetic
inertia), so the "what happens inside the turbine" story is visible: the de-loaded
over-speed reserve, the rotor deceleration and de-pitch that release it at the
event, and the shaft-torque swing that is the mechanical cost.

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\plot_freq_support_mechanics.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "results" / "em_interaction_sweep" / "freq_support_3way"
EVENT_TIME = 20.0
RATED_RPM = 7.56
T_LO, T_HI = 0.0, 60.0
SETTLE = 0.5   # drop the FMU priming glitch in the first samples (t < SETTLE)

CASES = [
    ("deload_none", "no support", "#d62728"),
    ("deload_droop", "droop", "#1f77b4"),
    ("deload_droop_inertia", "droop + inertia", "#2ca02c"),
]

# (column, axis label, optional reference line)
PANELS = [
    ("fmu_RotSpeed", "Rotor speed\n[rpm]", RATED_RPM),
    ("fmu_BldPitch1", "Blade pitch\n[deg]", None),
    ("fmu_GenTq", "Generator torque\n[kN$\\cdot$m]", None),
    ("fmu_HSShftTq", "Shaft torque\n[kN$\\cdot$m]", None),
]


def main() -> None:
    frames = {}
    for tag, _lab, _c in CASES:
        fp = DATA / f"{tag}.csv"
        if fp.exists():
            frames[tag] = pd.read_csv(fp)
    if not frames:
        print(f"No CSVs in {DATA}; run freq_support_3way.py first.")
        return

    panels = [p for p in PANELS
              if any(p[0] in df.columns for df in frames.values())]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.0, 2.2 * len(panels)),
                             sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, ylabel, ref) in zip(axes, panels):
        for tag, lab, colour in CASES:
            df = frames.get(tag)
            if df is None or col not in df.columns:
                continue
            m = (df["t"] >= max(T_LO, SETTLE)) & (df["t"] <= T_HI)
            ls = "--" if tag == "deload_none" else "-"
            ax.plot(df["t"][m], df[col][m], color=colour, lw=1.3, ls=ls,
                    label=lab)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=0.8)
            ax.text(T_HI, ref, " rated", va="bottom", ha="right",
                    fontsize=8, color="k")
        ax.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_xlim(T_LO, T_HI)
    axes[0].annotate("de-loaded\n(over-speed reserve)", xy=(15, RATED_RPM),
                     xytext=(11.5, RATED_RPM + 0.25), fontsize=8, color="#555")
    axes[0].legend(loc="lower right", fontsize=9, ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Turbine mechanics during a de-loaded gas-turbine-trip "
                 "frequency-support event\n(OpenFAST, Region 3)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = DATA / "freq_support_mechanics.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
