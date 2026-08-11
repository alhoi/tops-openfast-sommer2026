"""Compare the 5.3 Hz content of f_grid / P_uic before and after the X_q_t fix."""
import sys
import numpy as np
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\r3_fix_check.csv"
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
d = pd.read_csv(csv)
m = d["t"] >= lo
t = d["t"][m].values
dt = np.median(np.diff(t))


def hf(y):
    y = y - pd.Series(y).rolling(int(1.0 / dt), center=True, min_periods=1).mean().values
    y = y - np.mean(y)
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), dt)
    k = np.argmax(Y[1:]) + 1
    return float(np.std(y)), float(f[k]), float(2 * Y[k] / np.sum(np.hanning(len(y))))


print(f"file={csv}  window t>={lo}")
for c in ("f_grid_hz", "P_uic_bus_sys_pu", "fmu_GenTq"):
    if c in d.columns:
        s, fpk, amp = hf(d[c][m].values)
        print(f"  {c:20s} HF std={s:.4e}  dom {fpk:5.2f} Hz  amp={amp:.4e}")
print("\n(unfixed baseline: f_grid HF ~9.9e-4 @ 5.32 Hz, P_uic HF ~2.4e-3 @ 5.32 Hz)")
