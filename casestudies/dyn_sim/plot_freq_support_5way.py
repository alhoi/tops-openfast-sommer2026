"""Five-way frequency-support comparison on the GT-trip event.

Plots grid frequency and WT electrical power for five controller
configurations, to separate the droop / inertia contributions and show
what the support-path notch (tuned to the tower mode) costs on a step event:

  no support
  droop
  droop + notch
  droop + inertia
  droop + inertia + notch

Reads existing CSVs from the freq_support_3way results dir (no simulation).
Each case is individually best-tuned: the inertia cases use K_inertia 8e7
(freq-LPF 2 Hz); the notch cases use f0 = 0.233 Hz, Q = 8 (narrow, so it
nulls the tower mode with minimal off-band phase penalty on the step).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep" / "freq_support_3way"

EVENT_TIME = 20.0
LOAD_STEP_MW = 15.8

# (csv stem, label, colour, linestyle)
# Original set: inertia 5e7 / LPF 2.0, notch Q2.
CASES_ORIG = [
    ("deload_none",                  "no support",              "#7f7f7f", "--"),
    ("deload_droop",                 "droop",                   "#1f77b4", "-"),
    ("deload_droop_notch",           "droop + notch",           "#ff7f0e", "-"),
    ("deload_droop_inertia_orig5e7", "droop + inertia",         "#2ca02c", "-"),
    ("deload_droop_inertia_notch",   "droop + inertia + notch", "#9467bd", "-"),
]
# Best-tuned per case: inertia 8e7 (fast, low RoCoF), narrow notch Q8.
CASES_TUNED = [
    ("deload_none",              "no support",               "#7f7f7f", "--"),
    ("deload_droop",             "droop",                    "#1f77b4", "-"),
    ("deload_droop_q8",          "droop + notch",            "#ff7f0e", "-"),
    ("deload_droop_inertia_i8",  "droop + inertia",          "#2ca02c", "-"),
    ("deload_droop_inertia_q8",  "droop + inertia + notch",  "#9467bd", "-"),
]
# Over-rating cap: same cases and colours as the tuned set, but the inertia
# burst is capped (--max-over-nm -3e6) so peak power stays near rating.
CASES_CAP = [
    ("deload_none",                "no support",               "#7f7f7f", "--"),
    ("deload_droop",               "droop",                    "#1f77b4", "-"),
    ("deload_droop_q8",            "droop + notch",            "#ff7f0e", "-"),
    ("deload_droop_inertia_cap",   "droop + inertia",          "#2ca02c", "-"),
    ("deload_droop_inertia_capq8", "droop + inertia + notch",  "#9467bd", "-"),
]
# Same capped scenario but with UIC perfect_tracking on (isochronous internal
# frequency). Stems carry the _pt1 / _pt1q8 suffixes from the pt=1 reruns.
CASES_PT1 = [
    ("deload_none_pt1",             "no support",               "#7f7f7f", "--"),
    ("deload_droop_pt1",            "droop",                    "#1f77b4", "-"),
    ("deload_droop_pt1q8",          "droop + notch",            "#ff7f0e", "-"),
    ("deload_droop_inertia_pt1",    "droop + inertia",          "#2ca02c", "-"),
    ("deload_droop_inertia_pt1q8",  "droop + inertia + notch",  "#9467bd", "-"),
]


def metrics(df: pd.DataFrame) -> dict[str, float]:
    t = df["t"].to_numpy()
    f = df["f_grid_hz"].to_numpy()
    pre = f[(t >= EVENT_TIME - 5) & (t < EVENT_TIME)].mean()
    win = (t >= EVENT_TIME) & (t <= EVENT_TIME + 2.0)
    rocof = float(np.max(np.abs(np.gradient(f[win], t[win])))) * 1000.0
    nadir = float(np.min(f[t >= EVENT_TIME]) - pre) * 1000.0
    settled = float(f[t >= t[-1] - 8.0].mean() - pre) * 1000.0
    return {"rocof": rocof, "nadir": nadir, "settled": settled}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned", action="store_true",
                    help="Use the best-tuned per-case configs (inertia 8e7, "
                         "notch Q8) and write freq_support_5way_tuned.png.")
    ap.add_argument("--cap", action="store_true",
                    help="Show the over-rating cap comparison (droop+inertia "
                         "8e7 uncapped vs capped burst) -> _cap.png.")
    ap.add_argument("--pt1", action="store_true",
                    help="Capped scenario with UIC perfect_tracking on "
                         "(isochronous) -> _pt1.png.")
    args = ap.parse_args()

    if args.pt1:
        cases = CASES_PT1
        out_png = OUT_DIR / "freq_support_5way_pt1.png"
        note = "  (UIC perfect tracking)"
    elif args.cap:
        cases = CASES_CAP
        out_png = OUT_DIR / "freq_support_5way_cap.png"
        note = "  (over-rating burst cap)"
    elif args.tuned:
        cases = CASES_TUNED
        out_png = OUT_DIR / "freq_support_5way_tuned.png"
        note = "  (best-tuned)"
    else:
        cases = CASES_ORIG
        out_png = OUT_DIR / "freq_support_5way.png"
        note = ""

    frames = {}
    for stem, *_ in cases:
        fp = OUT_DIR / f"{stem}.csv"
        if fp.exists():
            frames[stem] = pd.read_csv(fp)
        else:
            print(f"  (missing {fp.name})")

    print(f"\n{'case':>26}  {'RoCoF [mHz/s]':>13}  {'nadir [mHz]':>12}  "
          f"{'settled [mHz]':>13}")
    for stem, label, *_ in cases:
        if stem in frames:
            m = metrics(frames[stem])
            print(f"{label:>26}  {m['rocof']:>13.1f}  {m['nadir']:>12.1f}  "
                  f"{m['settled']:>13.1f}")

    fig, (ax_f, ax_p) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    for stem, label, colour, ls in cases:
        if stem not in frames:
            continue
        df = frames[stem]
        t = df["t"].to_numpy()
        lw = 1.3 if ls == "--" else 1.5
        ax_f.plot(t, df["f_grid_hz"], color=colour, lw=lw, ls=ls, label=label)
        if "P_e_sys_pu" in df:
            ax_p.plot(t, 100.0 * df["P_e_sys_pu"], color=colour, lw=lw, ls=ls,
                      label=label)

    ax_f.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
    ax_p.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
    ax_p.axhline(15.0, color="k", ls="--", lw=0.9)
    ax_p.text(60, 15.0, " 15 MW rating", va="bottom", ha="right",
              fontsize=8, color="k")
    ax_f.set_ylabel("Grid frequency  $f_{grid}$  [Hz]")
    ax_p.set_ylabel("WT electrical power  [MW]")
    ax_p.set_xlabel("Time [s]")
    ax_f.set_title(f"Frequency support on a {LOAD_STEP_MW:.1f} MW gas-turbine "
                   f"trip: droop / inertia / notch (OpenFAST, Region 3){note}")
    for ax in (ax_f, ax_p):
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
