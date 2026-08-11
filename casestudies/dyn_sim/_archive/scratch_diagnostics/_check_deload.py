"""Check whether the standing de-load offset over-speeds the rotor and reduces
power (creates a reserve). Compares pre-deload (t<5s) vs settled (last 15s)."""
import sys
import numpy as np
import pandas as pd

_csv = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\deload_diag.csv"
d = pd.read_csv(_csv)
pre = d["t"] < 5
late = d["t"] >= d["t"].iloc[-1] - 15
for name, unit, sc in [
    ("fmu_RotSpeed", "rpm", 1.0),
    ("fmu_GenTq", "kNm", 1.0),
    ("fmu_BldPitch1", "deg", 1.0),
    ("P_uic_bus_sys_pu", "MW", 100.0),
    ("zmq_torque_offset_nm", "kNm", 1e-3),
    ("f_grid_hz", "Hz", 1.0),
]:
    if name in d.columns:
        a = d[name][pre].mean() * sc
        b = d[name][late].mean() * sc
        print(f"  {name:24s} pre={a:10.4f}  settled={b:10.4f}  d={b-a:+.4f} {unit}")
p0 = d["P_uic_bus_sys_pu"][pre].mean() * 100
p1 = d["P_uic_bus_sys_pu"][late].mean() * 100
print(f"\nReserve created (power drop): {p0 - p1:.3f} MW  ({(p0-p1)/p0*100:.1f} %)")
