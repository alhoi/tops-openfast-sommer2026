"""Settled-window (last 20 s) means for the GT-trip ON vs OFF runs, to see
whether the droop torque offset actually reaches the grid as extra power."""
import numpy as np
import pandas as pd

for tag, path in [("OFF", r"results\sweep\gt_trip_off.csv"),
                  ("ON ", r"results\sweep\gt_trip_on.csv")]:
    d = pd.read_csv(path)
    m = d["t"] >= d["t"].iloc[-1] - 20
    pre = d["t"] < 10
    off = d["zmq_torque_offset_nm"][m].mean() if "zmq_torque_offset_nm" in d else 0.0
    print(f"--- {tag} (settled last 20 s) ---")
    print(f"  f_grid        : {d['f_grid_hz'][m].mean():.4f} Hz")
    print(f"  ZMQ offset    : {off/1e3:8.1f} kNm (mean)")
    print(f"  GenTq         : {d['fmu_GenTq'][m].mean():8.1f} kNm  (pre-trip {d['fmu_GenTq'][pre].mean():.1f})")
    print(f"  RotSpeed      : {d['fmu_RotSpeed'][m].mean():.4f} rpm  (pre {d['fmu_RotSpeed'][pre].mean():.4f})")
    print(f"  P_uic (sys pu): {d['P_uic_bus_sys_pu'][m].mean():.5f}  = {d['P_uic_bus_sys_pu'][m].mean()*100:.3f} MW")
    print(f"  P_sync_gen    : {d['P_sync_generators_total_sys_pu'][m].mean()*100:.3f} MW")
