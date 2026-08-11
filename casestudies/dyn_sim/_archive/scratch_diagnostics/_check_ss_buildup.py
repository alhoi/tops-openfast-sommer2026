"""Print the tower-SS build-up envelope (sine-fit @0.234 Hz) over time windows."""
import sys
import numpy as np
import pandas as pd

F = 0.234
csv = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\zmqgrid_on_long.csv"
d = pd.read_csv(csv)
w = 2 * np.pi * F


def amp(lo, hi, col):
    m = (d["t"] >= lo) & (d["t"] <= hi)
    t = d["t"][m].values
    y = d[col][m].values - d[col][m].mean()
    A = np.c_[np.cos(w * t), np.sin(w * t)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.hypot(c[0], c[1]))


print(f"file={csv}")
print("SS build-up (fmu_YawBrTAyp amp @0.234 Hz, m/s2):")
for a, b in [(15, 45), (45, 75), (90, 120), (150, 180), (210, 240), (255, 300)]:
    print(f"  t={a:3d}-{b:3d}s : {amp(a, b, 'fmu_YawBrTAyp'):.4f}")
print(f"GenTq  amp late [255-300]: {amp(255, 300, 'fmu_GenTq'):.1f} kNm")
print(f"f_grid amp late [255-300]: {amp(255, 300, 'f_grid_hz')*1e3:.1f} mHz")
if "zmq_torque_offset_nm" in d.columns:
    print(f"offset amp late [255-300]: {amp(255, 300, 'zmq_torque_offset_nm')*1e-3:.1f} kNm")
