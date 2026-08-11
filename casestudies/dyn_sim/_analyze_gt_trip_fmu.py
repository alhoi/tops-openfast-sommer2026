"""GT-trip (generation-deficit step) comparison: frequency support ON vs OFF.

A permanent load step on the main gas-turbine bus emulates a gas-turbine trip.
With the frequency-support coupling ON the wind turbine's droop injects extra
power on the frequency dip (reducing the nadir / settled dip) at the cost of a
generator-torque transient that kicks the tower side-to-side mode (which then
rings down at 0.234 Hz - a broadband event, NOT resonant build-up).

Reports grid-frequency baseline / nadir / settled dip and the tower response,
ON vs OFF, and makes a 3-panel figure.

Usage: python _analyze_gt_trip_fmu.py [on_csv] [off_csv] [event_time] [out_png]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
on_csv = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\gt_trip_on.csv"
off_csv = sys.argv[2] if len(sys.argv) > 2 else r"results\sweep\gt_trip_off.csv"
t_ev = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
out_png = sys.argv[4] if len(sys.argv) > 4 else \
    str(PROJECT_ROOT / "results" / "em_interaction_sweep" / "gt_trip_fmu.png")

on = pd.read_csv(on_csv)
off = pd.read_csv(off_csv)
t_end = float(on["t"].iloc[-1])


def summary(d, label):
    t = d["t"].values
    f = d["f_grid_hz"].values
    pre = f[(t >= 2.0) & (t < t_ev)]
    base = float(pre.mean())
    post = (t >= t_ev)
    nadir = float(f[post].min())
    settled = float(f[t >= t_end - 15].mean())
    ss = d["fmu_YawBrTAyp"].values
    ss_post = ss[post]
    ss_peak = float(np.max(np.abs(ss_post - ss_post.mean())))
    ss_rms = float(np.std(ss_post))
    gt = d["fmu_GenTq"].values
    gt_peak = float(np.max(np.abs(gt[post] - gt[t < t_ev].mean())))
    print(f"--- {label} ---")
    print(f"  baseline f      : {base:.4f} Hz")
    print(f"  nadir f         : {nadir:.4f} Hz   (dip {(base-nadir)*1e3:6.1f} mHz)")
    print(f"  settled f       : {settled:.4f} Hz   (dip {(base-settled)*1e3:6.1f} mHz)")
    print(f"  tower SS peak   : {ss_peak:.4f} m/s2   RMS {ss_rms:.4f}")
    print(f"  GenTq peak dev  : {gt_peak:.1f} kNm")
    return dict(base=base, nadir=nadir, settled=settled,
                ss_peak=ss_peak, gt_peak=gt_peak)


s_off = summary(off, "UTEN frekvensstøtte (droop av)")
s_on = summary(on, "MED frekvensstøtte (droop på)")
print("\n=== support effect (ON vs OFF) ===")
print(f"  nadir dip   : {(s_off['base']-s_off['nadir'])*1e3:.1f} -> "
      f"{(s_on['base']-s_on['nadir'])*1e3:.1f} mHz")
print(f"  settled dip : {(s_off['base']-s_off['settled'])*1e3:.1f} -> "
      f"{(s_on['base']-s_on['settled'])*1e3:.1f} mHz")
print(f"  tower SS pk : {s_off['ss_peak']:.4f} -> {s_on['ss_peak']:.4f} m/s2 "
      f"(x{s_on['ss_peak']/s_off['ss_peak']:.2f})")

# ---------------------------------------------------------------- figure
C_OFF, C_ON = "0.45", "#c1272d"
fig, ax = plt.subplots(5, 1, figsize=(9, 12), sharex=True)

# (0) Grid frequency - the grid effect of the support.
ax[0].plot(off["t"], off["f_grid_hz"], color=C_OFF, lw=1.0,
           label="uten frekvensstøtte")
ax[0].plot(on["t"], on["f_grid_hz"], color=C_ON, lw=1.0,
           label="med frekvensstøtte")
ax[0].set_ylabel("Nettfrekvens\n[Hz]")
ax[0].legend(loc="upper right", fontsize=8)
ax[0].set_title("GT-utfall (permanent generasjonsunderskudd): frekvensstøtte av/på")

# (1) WT active power - the reserve released into the grid.
ax[1].plot(off["t"], off["P_uic_bus_sys_pu"] * 100.0, color=C_OFF, lw=0.9)
ax[1].plot(on["t"], on["P_uic_bus_sys_pu"] * 100.0, color=C_ON, lw=0.9)
ax[1].set_ylabel("WT aktiv effekt\n[MW]")

# (2) Generator torque - the actuation.
gt0 = float(on["fmu_GenTq"][on["t"] < t_ev].mean())
ax[2].plot(off["t"], off["fmu_GenTq"] - gt0, color=C_OFF, lw=0.9)
ax[2].plot(on["t"], on["fmu_GenTq"] - gt0, color=C_ON, lw=0.9)
ax[2].set_ylabel("Generatormoment\nΔT [kNm]")

# (3) Blade pitch - the de-load / reserve-release mechanism (Region 3).
if "fmu_BldPitch1" in on.columns:
    ax[3].plot(off["t"], off["fmu_BldPitch1"], color=C_OFF, lw=0.9)
    ax[3].plot(on["t"], on["fmu_BldPitch1"], color=C_ON, lw=0.9)
ax[3].set_ylabel("Bladvinkel\n[°]")

# (4) Tower side-to-side acceleration - the mechanical cost.
ax[4].plot(off["t"], off["fmu_YawBrTAyp"], color=C_OFF, lw=0.9)
ax[4].plot(on["t"], on["fmu_YawBrTAyp"], color=C_ON, lw=0.9)
ax[4].set_ylabel("Tårn side-til-side\n[m/s²]")
ax[4].set_xlabel("Tid [s]")

for a in ax:
    a.grid(True, alpha=0.3)
    a.axvline(t_ev, color="k", ls=":", lw=0.8)

fig.tight_layout()
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=130)
print(f"\nwrote {out_png}")
