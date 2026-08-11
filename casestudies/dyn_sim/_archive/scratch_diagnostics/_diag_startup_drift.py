"""Diagnose the startup grid-frequency drift in the 2WT LEOGO run.

Prints, at selected times before the isolation step, the grid frequency, the
two turbine bus powers, total synchronous generation, and the rotor states, so
we can see whether a small initialisation power imbalance is what drifts the
centre-of-inertia frequency away from 50 Hz.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SYS_MVA = 100.0

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    'casestudies/dyn_sim/results/WT_LEOGO_2wt_results.csv')
df = pd.read_csv(csv_path)

times = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 34.0]

cols = {
    'f_hz': 'grid_freq_hz',
    'P_wt1_MW': 'P_uic_bus_sys_pu_wt1',
    'P_wt2_MW': 'P_uic_bus_sys_pu_wt2',
    'P_gen_MW': 'P_sync_generators_total_sys_pu',
    'w_m1': 'omega_m_pu_wt1',
    'w_m2': 'omega_m_pu_wt2',
    'pitch1': 'pitch_deg_wt1',
    'Paero1_MW': 'P_aero_sys_pu_wt1',
    'Paero2_MW': 'P_aero_sys_pu_wt2',
}

print(f"csv: {csv_path.name}  rows={len(df)}")
print(f"columns present: {[c for c in cols.values() if c in df.columns]}")
print()

t_arr = df['t'].to_numpy()
hdr = f"{'t[s]':>6} {'f[Hz]':>9} {'Pwt1':>8} {'Pwt2':>8} {'Pgen':>8} " \
      f"{'w_m1':>8} {'w_m2':>8} {'pitch1':>7} {'Paero1':>8} {'Paero2':>8}"
print(hdr)
for tt in times:
    i = int(np.argmin(np.abs(t_arr - tt)))
    r = df.iloc[i]
    def mw(col):
        return r[col] * SYS_MVA if col in df.columns else float('nan')
    print(f"{t_arr[i]:6.2f} {r['grid_freq_hz']:9.5f} "
          f"{mw('P_uic_bus_sys_pu_wt1'):8.3f} {mw('P_uic_bus_sys_pu_wt2'):8.3f} "
          f"{mw('P_sync_generators_total_sys_pu'):8.3f} "
          f"{r.get('omega_m_pu_wt1', float('nan')):8.5f} "
          f"{r.get('omega_m_pu_wt2', float('nan')):8.5f} "
          f"{r.get('pitch_deg_wt1', float('nan')):7.3f} "
          f"{mw('P_aero_sys_pu_wt1'):8.3f} {mw('P_aero_sys_pu_wt2'):8.3f}")

# Net imbalance proxy: total electrical injection vs an implied constant load.
print()
i0 = int(np.argmin(np.abs(t_arr - 0.0)))
iN = int(np.argmin(np.abs(t_arr - 34.0)))
for label, i in [('t=0', i0), ('t=34', iN)]:
    p_inj = 0.0
    for c in ('P_uic_bus_sys_pu_wt1', 'P_uic_bus_sys_pu_wt2',
              'P_sync_generators_total_sys_pu'):
        if c in df.columns:
            p_inj += df.iloc[i][c] * SYS_MVA
    print(f"{label}: total electrical injection (WT1+WT2+gens) = {p_inj:8.3f} MW")
