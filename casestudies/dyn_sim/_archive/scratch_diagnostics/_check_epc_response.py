"""Check whether the ROSCO closed-loop generator torque responds to the
ElecPwrCom (demanded electrical power) modulation -- i.e. whether the
electrical demand actually reaches the OpenFAST generator ("path A").

Reads the path-A diagnostic CSV, sine-fits the generator torque and the
tower side-to-side acceleration at the modulation frequency, and compares
the torque response against the imposed ElecPwrCom command amplitude.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV = (Path(sys.argv[1]) if len(sys.argv) > 1
       else PROJECT_ROOT / "results" / "sweep" / "diag_epc.csv")

F_MOD = 0.234      # ElecPwrCom modulation frequency [Hz]
ONSET = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0


def sine_amp(t, y, f):
    w = 2.0 * np.pi * f
    A = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.hypot(c[0], c[1]))


df = pd.read_csv(CSV)
t = df["t"].to_numpy()
m = t >= ONSET
tt = t[m]

gentq = df["fmu_GenTq"].to_numpy()[m]          # kNm
ss = df["fmu_YawBrTAyp"].to_numpy()[m]         # m/s^2
epc = df["elecpwr_mod_factor"].to_numpy()[m]   # command factor (1 +/- amp*sin)

gentq_mean = float(np.mean(gentq))
gentq_amp = sine_amp(tt, gentq - gentq_mean, F_MOD)
ss_amp = sine_amp(tt, ss - np.mean(ss), F_MOD)
epc_amp = sine_amp(tt, epc - np.mean(epc), F_MOD)

print(f"ElecPwrCom command modulation amplitude : {epc_amp*100:.1f} % of mean")
print(f"GenTq mean = {gentq_mean:.1f} kNm")
print(f"GenTq response @ {F_MOD} Hz (sine-fit)  : {gentq_amp:.1f} kNm "
      f"({gentq_amp/max(abs(gentq_mean),1e-9)*100:.2f} % of mean)")
print(f"Tower SS accel @ {F_MOD} Hz (sine-fit)  : {ss_amp:.4g} m/s^2")
print(f"GenTq total std (post-onset)            : {np.std(gentq):.1f} kNm, "
      f"peak-to-peak {np.ptp(gentq):.1f} kNm")

verdict = ("RESPONDS -> ElecPwrCom reaches the generator (path A works)"
           if gentq_amp > 0.005 * abs(gentq_mean)
           else "FLAT -> ElecPwrCom is IGNORED by ROSCO (path A blocked)")
print(f"VERDICT: {verdict}")
