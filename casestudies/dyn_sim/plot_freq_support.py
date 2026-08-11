"""Region-3 frequency-support demonstration: a load step with the WT
virtual-inertia + droop support OFF vs ON.

Companion to plot_r3_resonance.py. Where the resonance figure shows the RISK
of the electromechanical frequency-support coupling (a process-load pulsation
at the tower mode drives a resonant tower response), this figure shows the
BENEFIT: for a plain load step the same droop + virtual-inertia support
arrests the rate of change of frequency, reduces the nadir, and lifts the
settled frequency - while the tower stays calm (the step does not sit on the
0.234 Hz side-to-side mode).

Reads two co-sim CSVs that share every parameter except the support gains:
  * OFF : --droop-nm-per-hz 0  --inertia-nm-s-per-hz 0
  * ON  : --droop-nm-per-hz 1e7 --inertia-nm-s-per-hz 5e6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

C_ON = "#0b6e4f"    # green - support on
C_OFF = "#8a8f98"   # grey  - support off
C_LOAD = "#2c3e50"


def metrics(df: pd.DataFrame, tev: float):
    t = df["t"].to_numpy(float)
    f = df["f_grid_hz"].to_numpy(float)
    base = float(f[(t >= tev - 10) & (t < tev)].mean())
    post = (t >= tev) & (t <= tev + 30)
    nadir = float(f[post].min()) if post.any() else base
    settled = float(f[t >= t[-1] - 20].mean())
    m = (t >= tev) & (t <= tev + 3)
    rocof = float(np.max(np.abs(np.gradient(f[m], t[m])))) if m.sum() > 2 else 0.0
    return base, nadir, settled, rocof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--on-csv", default=r"results\sweep\fsupport_on.csv")
    p.add_argument("--off-csv", default=r"results\sweep\fsupport_off.csv")
    p.add_argument("--event-time", type=float, default=40.0)
    p.add_argument("--load-step-mw", type=float, default=4.0)
    p.add_argument("--out", default=r"results\em_interaction_sweep\r3_freq_support.png")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    on = pd.read_csv(args.on_csv)
    off = pd.read_csv(args.off_csv)
    tev = args.event_time

    b_on, nad_on, set_on, roc_on = metrics(on, tev)
    b_off, nad_off, set_off, roc_off = metrics(off, tev)

    # Deviations from the pre-event baseline (mHz), positive = dip depth.
    nadir_dip_on = (b_on - nad_on) * 1e3
    nadir_dip_off = (b_off - nad_off) * 1e3
    settled_dip_on = (b_on - set_on) * 1e3
    settled_dip_off = (b_off - set_off) * 1e3

    p_on = on["P_uic_bus_sys_pu"].to_numpy(float) * 100.0
    p_off = off["P_uic_bus_sys_pu"].to_numpy(float) * 100.0
    boost = (on["P_uic_bus_sys_pu"][on["t"] >= on["t"].iloc[-1] - 20].mean()
             - off["P_uic_bus_sys_pu"][off["t"] >= off["t"].iloc[-1] - 20].mean()) * 100.0

    ss_on = on["fmu_BldPitch1"].to_numpy(float)
    ss_off = off["fmu_BldPitch1"].to_numpy(float)

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.3, 1.0, 1.0], hspace=0.33)

    t_on = on["t"].to_numpy(float)
    t_off = off["t"].to_numpy(float)
    t_end = float(t_on[-1])
    x0 = tev - 10.0

    # --- (1) grid frequency ----------------------------------------------------
    ax_f = fig.add_subplot(gs[0])
    ax_f.plot(t_off, off["f_grid_hz"], color=C_OFF, lw=1.4,
              label=f"støtte AV  (nadir −{nadir_dip_off:.0f} mHz, stasj. −{settled_dip_off:.0f} mHz)")
    ax_f.plot(t_on, on["f_grid_hz"], color=C_ON, lw=1.6,
              label=f"støtte PÅ  (nadir −{nadir_dip_on:.0f} mHz, stasj. −{settled_dip_on:.0f} mHz)")
    ax_f.axvline(tev, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_f.axhline(b_on, color="k", ls="--", lw=0.6, alpha=0.4)
    ax_f.annotate(f"+{args.load_step_mw:.0f} MW last", xy=(tev, ax_f.get_ylim()[0]),
                  xytext=(tev + 2, b_on - 0.9 * (b_on - min(nad_on, nad_off))),
                  fontsize=9, color="k")
    ax_f.set_ylabel("Nettfrekvens [Hz]")
    ax_f.set_xlim(x0, t_end)
    ax_f.legend(loc="lower right", framealpha=0.9)
    ax_f.set_title(
        f"Frekvensstøtte ved +{args.load_step_mw:.0f} MW laststeg:  "
        f"RoCoF {roc_off*1e3:.0f} → {roc_on*1e3:.0f} mHz/s,  "
        f"nadir −{nadir_dip_off:.0f} → −{nadir_dip_on:.0f} mHz", fontsize=11)

    # --- (2) WT electrical power -----------------------------------------------
    ax_p = fig.add_subplot(gs[1])
    ax_p.plot(t_off, p_off, color=C_OFF, lw=1.3, label="støtte AV")
    ax_p.plot(t_on, p_on, color=C_ON, lw=1.5, label=f"støtte PÅ  (+{boost:.2f} MW vedvarende)")
    ax_p.axvline(tev, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_p.set_ylabel("WT-effekt\nP_uic [MW]")
    ax_p.set_xlim(x0, t_end)
    ax_p.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax_p.set_title("Vindturbinen leverer støtteeffekt (KE + pitch-reserve)", fontsize=11)

    # --- (3) blade pitch: the sustained-reserve mechanism ----------------------
    ax_s = fig.add_subplot(gs[2])
    ax_s.plot(t_off, ss_off, color=C_OFF, lw=1.3, label="støtte AV")
    ax_s.plot(t_on, ss_on, color=C_ON, lw=1.5, label="støtte PÅ")
    ax_s.axvline(tev, color="k", ls=":", lw=0.8, alpha=0.6)
    ax_s.set_ylabel("Bladvinkel\n[grader]")
    ax_s.set_xlabel("Tid [s]")
    ax_s.set_xlim(x0, t_end)
    ax_s.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_s.set_title("ROSCO pitcher for å frigjøre vedvarende reserve (de-load)",
                   fontsize=11)

    fig.suptitle(
        "OpenFAST  —  Region 3 (13 m/s): frekvensstøtte (droop + virtuell treghet) demper "
        "laststeget\n"
        "samme E↔M-kobling som gir tårnresonans ved 0.234 Hz gir her god "
        "nettstøtte",
        fontsize=12.5, y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Lagret figur: {out}")
    print(f"  baseline f: on={b_on:.4f}  off={b_off:.4f} Hz")
    print(f"  nadir dip : on=-{nadir_dip_on:.1f}  off=-{nadir_dip_off:.1f} mHz")
    print(f"  settled dip: on=-{settled_dip_on:.1f}  off=-{settled_dip_off:.1f} mHz")
    print(f"  RoCoF     : on={roc_on*1e3:.1f}  off={roc_off*1e3:.1f} mHz/s")
    print(f"  WT support boost = {boost:.3f} MW")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
