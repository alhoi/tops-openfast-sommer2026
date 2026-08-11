"""Compare settled + transient power for the two de-loaded GT-trip runs to see
whether the released reserve becomes SUSTAINED grid power or is only transient
rotor kinetic energy."""
import numpy as np
import pandas as pd

t_ev = 50.0
files = [("OFF (no release)", r"results\sweep\r3_gt_off.csv"),
         ("ON  (release)   ", r"results\sweep\r3_gt_on2.csv")]
for tag, path in files:
    d = pd.read_csv(path)
    pre = (d["t"] >= t_ev - 10) & (d["t"] < t_ev)         # curtailed, pre-trip
    early = (d["t"] >= t_ev) & (d["t"] < t_ev + 8)         # transient after trip
    late = d["t"] >= d["t"].iloc[-1] - 15                  # settled
    off = d["zmq_torque_offset_nm"]
    P = d["P_uic_bus_sys_pu"] * 100
    print(f"--- {tag} ---")
    print(f"  offset  pre={off[pre].mean()/1e3:8.1f}  early={off[early].mean()/1e3:8.1f}  late={off[late].mean()/1e3:8.1f} kNm")
    print(f"  P_uic   pre={P[pre].mean():7.3f}  early_pk={P[early].max():7.3f}  late={P[late].mean():7.3f} MW")
    print(f"  RotSpd  pre={d['fmu_RotSpeed'][pre].mean():.4f}  late={d['fmu_RotSpeed'][late].mean():.4f} rpm")

on = pd.read_csv(files[1][1]); off_d = pd.read_csv(files[0][1])
late = on["t"] >= on["t"].iloc[-1] - 15
dP = (on["P_uic_bus_sys_pu"][late].mean() - off_d["P_uic_bus_sys_pu"][late].mean()) * 100
print(f"\nSustained extra power ON vs OFF (settled): {dP:+.3f} MW")
