"""Estimate the tower side-to-side ring-down decay constant from FMU data.

Reads the on-resonance ring-down run, isolates the FREE-decay window after the
excitation has fully ramped off, extracts the analytic (Hilbert) envelope of the
SS acceleration, and fits ln(envelope) = ln(A0) - t/tau to recover the decay
time constant tau, damping ratio zeta = 1/(omega_n*tau) and quality factor
Q = 1/(2*zeta). Purely diagnostic (no fabricated numbers).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV = PROJECT_ROOT / "results" / "sweep" / "ss_1turb_0p234_ringdown.csv"

F_SS = 0.234            # tower SS eigenfrequency [Hz]
STOP = 200.0           # excitation switched off
RAMP = 15.0            # smoothstep ramp-off duration [s]
FREE_LO = STOP + RAMP  # start of genuine free decay [s]

df = pd.read_csv(CSV)
t = df["t"].to_numpy()
y = df["fmu_YawBrTAyp"].to_numpy()

m = t >= FREE_LO
tt = t[m]
yy = y[m] - np.mean(y[m])

env = np.abs(hilbert(yy))
# Fit ln(env) vs t; weight by envelope to de-emphasise near-zero noise floor.
good = env > 0.15 * env.max()
p = np.polyfit(tt[good], np.log(env[good]), 1, w=env[good])
tau = -1.0 / p[0]

omega_n = 2.0 * np.pi * F_SS
zeta = 1.0 / (omega_n * tau)
Q = 1.0 / (2.0 * zeta)
n_cycles_1e = tau * F_SS

print(f"Free-decay window: {FREE_LO:.0f}-{tt[-1]:.0f} s "
      f"({(tt[-1]-FREE_LO)*F_SS:.1f} cycles)")
print(f"  tau (envelope 1/e)   = {tau:.0f} s")
print(f"  zeta (damping ratio) = {zeta*100:.2f} %")
print(f"  Q  = 1/(2*zeta)      = {Q:.0f}")
print(f"  cycles to decay 1/e  = {n_cycles_1e:.0f}")
print(f"  time to decay to 5%  = {3.0*tau:.0f} s (~3 tau)")
print(f"  env start {env[good][0]:.3g} -> end {env[good][-1]:.3g} m/s^2 "
      f"({env[good][-1]/env[good][0]*100:.0f}% over "
      f"{tt[good][-1]-tt[good][0]:.0f} s)")
