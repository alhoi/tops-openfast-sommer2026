"""Verify the grid-driven torque coupling: does the torque offset track the
live grid frequency via the droop law dT = -K_droop*(f_grid - f_nom)?

Regresses the logged torque offset on the grid-frequency deviation and reports
the fitted droop gain (should match the configured value) and correlation.
"""
import sys
import numpy as np
import pandas as pd

log = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\grid_zmq_log.csv"
t_lo = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
f_nom = 50.0

d = pd.read_csv(log)
d = d[np.isfinite(d["t_rosco"]) & (d["t_rosco"] >= t_lo)]
df = d["f_grid_hz"].to_numpy() - f_nom
off = d["torque_offset_nm"].to_numpy()

# Only regress where the offset is active (non-zero).
active = np.abs(off) > 1.0
df_a, off_a = df[active], off[active]

if df_a.size > 10 and np.std(df_a) > 0:
    # offset = slope * df + b ; expect slope ~ -K_droop
    slope, b = np.polyfit(df_a, off_a, 1)
    corr = np.corrcoef(df_a, off_a)[0, 1]
    print(f"file={log}  window t>={t_lo}s  n_active={df_a.size}")
    print(f"  grid df range          : [{df.min()*1e3:+.2f}, {df.max()*1e3:+.2f}] mHz")
    print(f"  torque offset range    : [{off.min():+.3e}, {off.max():+.3e}] Nm")
    print(f"  fitted droop slope      : {slope:.4e} Nm/Hz  (configured -2.0e7)")
    print(f"  correlation(offset, df) : {corr:+.4f}   (expect ~ -1: offset opposes df)")
else:
    print("Not enough active-offset variation to regress.")
    print(f"  n_active={df_a.size}  std(df_active)={np.std(df_a):.3e}")
