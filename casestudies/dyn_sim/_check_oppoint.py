"""Print the settled operating point (last 10 s) for a Region-3 check run."""
import sys
import numpy as np
import pandas as pd

csv = sys.argv[1] if len(sys.argv) > 1 else r"results\sweep\r3_check.csv"
d = pd.read_csv(csv)
late = d["t"] >= d["t"].iloc[-1] - 10
print(f"file={csv}  settled (last 10 s):")
for name, unit, sc in [
    ("fmu_Wind1VelX", "m/s", 1.0),
    ("fmu_RotSpeed", "rpm", 1.0),
    ("fmu_GenSpeed", "rpm", 1.0),
    ("fmu_BldPitch1", "deg", 1.0),
    ("fmu_GenTq", "kNm", 1.0),
    ("P_uic_bus_sys_pu", "MW", 100.0),
    ("f_grid_hz", "Hz", 1.0),
    ("fmu_YawBrTAyp", "m/s2 (std)", 1.0),
]:
    if name in d.columns:
        if "std" in unit:
            v = d[name][late].std() * sc
        else:
            v = d[name][late].mean() * sc
        print(f"  {name:20s} = {v:10.4f} {unit}")
