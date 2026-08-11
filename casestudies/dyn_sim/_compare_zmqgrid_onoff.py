"""ON/OFF comparison of the genuine grid-driven electromechanical coupling.

Two matched co-simulations differ ONLY in the frequency-support droop gain:
  ON  : --droop-nm-per-hz 2e7  (grid frequency drives generator torque via ZMQ)
  OFF : --droop-nm-per-hz 0    (identical code path, torque offset stays zero)

Both are driven by the SAME LEOGO disturbance: a zero-mean +/-3 MW process-load
oscillation at 0.234 Hz injected as a shunt on the main gas-turbine bus.

The comparison isolates the genuine electrical->mechanical interaction: with the
coupling ON the grid-frequency oscillation reaches the generator torque and the
tower side-to-side mode; with it OFF only the weak direct grid->UIC->drivetrain
path remains. It also shows the reciprocal leg (the turbine's torque modulation
flowing back into the grid frequency / genset power).

Usage:
    python _compare_zmqgrid_onoff.py [on_csv] [off_csv] [onset_s] [out_png]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = 0.234
PROJECT_ROOT = Path(__file__).resolve().parents[2]

on_csv = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\zmqgrid_on.csv"
off_csv = sys.argv[2] if len(sys.argv) > 2 else r"results\sweep\zmqgrid_off.csv"
onset = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
out_png = sys.argv[4] if len(sys.argv) > 4 else \
    str(PROJECT_ROOT / "results" / "em_interaction_sweep" / "zmqgrid_onoff.png")


def sine_amp(t, y, f, lo, hi):
    m = (t >= lo) & (t <= hi)
    tt = np.asarray(t[m], float)
    yy = np.asarray(y[m], float)
    yy = yy - yy.mean()
    w = 2 * np.pi * f
    A = np.c_[np.cos(w * tt), np.sin(w * tt)]
    c, *_ = np.linalg.lstsq(A, yy, rcond=None)
    return float(np.hypot(c[0], c[1]))


on = pd.read_csv(on_csv)
off = pd.read_csv(off_csv)
t_end = float(on["t"].iloc[-1])
lo_late, hi_late = 0.6 * t_end, t_end          # settled/late window
lo_all, hi_all = onset, t_end

print(f"ON : {on_csv}")
print(f"OFF: {off_csv}")
print(f"late window [{lo_late:.0f}, {hi_late:.0f}] s  (amplitudes @ {F} Hz)\n")

rows = [
    ("f_grid_hz", "Hz", 1.0),
    ("zmq_torque_offset_nm", "kNm", 1e-3),
    ("fmu_GenTq", "kNm", 1.0),
    ("fmu_HSShftTq", "kNm", 1.0),
    ("fmu_YawBrTAyp", "m/s2", 1.0),
    ("P_uic_bus_sys_pu", "MW", 100.0),
    ("P_sync_generators_total_sys_pu", "MW", 100.0),
]
print(f"{'signal':32s} {'OFF':>12s} {'ON':>12s} {'ON/OFF':>8s}")
for c, unit, sc in rows:
    if c not in on.columns:
        continue
    a_off = sine_amp(off["t"], off[c], F, lo_late, hi_late) * sc
    a_on = sine_amp(on["t"], on[c], F, lo_late, hi_late) * sc
    ratio = a_on / a_off if a_off > 1e-12 else float("inf")
    print(f"{c:32s} {a_off:12.4e} {a_on:12.4e} {ratio:8.2f}  [{unit}]")

# ---------------------------------------------------------------- figure
C_OFF, C_ON = "0.45", "#c1272d"
fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

# (0) LEOGO process-load oscillation (the disturbance), same in both runs.
load_mw = 3.0 * on["load_step_scale"]
ax[0].plot(on["t"], load_mw, color="#1f5fb0", lw=1.0)
ax[0].set_ylabel("LEOGO-last\n[MW]")
ax[0].set_title("Prosesslast-oscillasjon på hovedbussen (±3 MW @ 0.234 Hz)")

# (1) Grid frequency response (reciprocal + support effect).
ax[1].plot(off["t"], (off["f_grid_hz"] - 50.0) * 1e3, color=C_OFF, lw=0.9,
           label="uten kobling (droop av)")
ax[1].plot(on["t"], (on["f_grid_hz"] - 50.0) * 1e3, color=C_ON, lw=0.9,
           label="med kobling (droop på)")
ax[1].set_ylabel("Nettfrekvens\nΔf [mHz]")
ax[1].legend(loc="upper right", fontsize=8)

# (2) Generator torque - the electromechanical actuation.
gt0 = on["fmu_GenTq"].iloc[0]
ax[2].plot(off["t"], off["fmu_GenTq"] - gt0, color=C_OFF, lw=0.9)
ax[2].plot(on["t"], on["fmu_GenTq"] - gt0, color=C_ON, lw=0.9)
ax[2].set_ylabel("Generatormoment\nΔT [kNm]")

# (3) Tower side-to-side acceleration - the mechanical response.
ax[3].plot(off["t"], off["fmu_YawBrTAyp"], color=C_OFF, lw=0.9)
ax[3].plot(on["t"], on["fmu_YawBrTAyp"], color=C_ON, lw=0.9)
ax[3].set_ylabel("Tårn side-til-side\n[m/s²]")
ax[3].set_xlabel("Tid [s]")

for a in ax:
    a.grid(True, alpha=0.3)
    a.axvline(10.0, color="k", ls=":", lw=0.8)

fig.suptitle(
    "Genuin elektromekanisk interaksjon LEOGO ↔ vindturbin\n"
    "samme nett-forstyrrelse, kun frekvensstøtte-kobling skrudd av/på",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=130)
print(f"\nwrote {out_png}")
