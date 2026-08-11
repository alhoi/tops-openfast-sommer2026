"""Check the GRID-side response at the tower-SS frequency in the ZMQ co-sim run.
Fits a 0.234 Hz sinusoid to grid-side signals to see whether the turbine's
torque modulation propagates INTO the LEOGO grid (turbine -> grid, the M->E
direction) even though the forcing is a prescribed clock sinusoid.
"""
import sys
import numpy as np
import pandas as pd

F = 0.234
csv = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\diag_zmq_torque.csv"
onset = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0


def sine_fit(t, y, f):
    t = np.asarray(t, float)
    y = np.asarray(y, float) - np.mean(y)
    w = 2 * np.pi * f
    A = np.c_[np.cos(w * t), np.sin(w * t)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.hypot(c[0], c[1]))


d = pd.read_csv(csv)
m = d["t"] >= onset
print(f"file={csv}  window t>={onset}s  n={m.sum()}\n")
cols = [
    ("f_grid_hz", "Hz"),
    ("omega_grid_pu", "pu"),
    ("V_WTG1_LV_pu", "pu"),
    ("P_uic_bus_sys_pu", "pu(100MVA)"),
    ("Q_uic_bus_sys_pu", "pu(100MVA)"),
    ("P_sync_generators_total_sys_pu", "pu(100MVA)"),
    ("fmu_GenTq", "kNm"),
    ("fmu_YawBrTAyp", "m/s2"),
]
for c, unit in cols:
    if c in d.columns:
        amp = sine_fit(d["t"][m], d[c][m], F)
        print(f"  {c:34s} mean={d[c][m].mean():+.5e}  amp@{F}Hz={amp:.4e} {unit}")
