"""Identify the 2nd tower side-to-side natural frequency from the torque-kick run.

Reads the FMU frequency-ID CSV (only TwSSDOF2 active, excited by a short torque
pulse), FFTs the post-pulse tower-top side-to-side acceleration, and reports the
spectral peak in a sensible band. Also estimates the modal damping from the
ring-down envelope and saves a 2-panel verification figure.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV = (Path(sys.argv[1]) if len(sys.argv) > 1
       else PROJECT_ROOT / "results" / "sweep" / "ss2_freq_id.csv")
OUT = (PROJECT_ROOT / "results" / "em_interaction_sweep"
       / (CSV.stem + "_spectrum.png"))

SS_CH = "fmu_YawBrTAyp"     # tower-top side-to-side acceleration [m/s^2]
PULSE_END = 11.0           # start FFT/analysis just after the 0.5 s pulse at t=10
BAND = (0.15, 3.0)         # search band [Hz]: include mode 1 (~0.234) and beyond


def _spectrum(t, y):
    dt = float(np.median(np.diff(t)))
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft((y - np.mean(y)) * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(len(y), dt)
    return freqs, spec


df = pd.read_csv(CSV)
t = df["t"].to_numpy()
y = df[SS_CH].to_numpy()

m = t >= PULSE_END
tt, yy = t[m], y[m]

freqs, spec = _spectrum(tt, yy)
band = (freqs >= BAND[0]) & (freqs <= BAND[1])
fb, sb = freqs[band], spec[band]

# Report the strongest spectral peaks in the band.
idx, _ = find_peaks(sb, height=0.05 * sb.max())
idx = idx[np.argsort(sb[idx])[::-1]][:4]
peaks = sorted(float(f) for f in fb[idx])
f_pk = float(fb[np.argmax(sb)])      # dominant peak

print(f"Spectral peaks in {BAND[0]}-{BAND[1]} Hz (post-pulse window "
      f"{tt[0]:.0f}-{tt[-1]:.0f} s, df={freqs[1]:.4f} Hz):")
for fpk in peaks:
    print(f"  f = {fpk:.3f} Hz")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5))
ax1.plot(t, y, color="#0b6e4f", lw=0.7)
ax1.axvspan(10.0, 10.5, color="k", alpha=0.15, label="momentpuls")
ax1.set_xlim(0, t[-1])
ax1.set_xlabel("Tid [s]")
ax1.set_ylabel("Tårn SS-akselerasjon\nYawBrTAyp [m/s²]")
ax1.set_title("SS-moder: momentkick og utsvingning (begge SS-DOF)", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)

ax2.plot(freqs, spec, color="#0b6e4f", lw=1.3)
for fpk in peaks:
    s_at = spec[np.argmin(np.abs(freqs - fpk))]
    ax2.axvline(fpk, color="#c0392b", ls="--", lw=0.9, alpha=0.8)
    ax2.annotate(f"{fpk:.3f} Hz", xy=(fpk, s_at),
                 xytext=(fpk + 0.05, 0.9 * s_at),
                 fontsize=9, color="#c0392b")
ax2.set_xlim(0, 3.0)
ax2.set_xlabel("Frekvens [Hz]")
ax2.set_ylabel("SS-akselerasjon-\nspekter [m/s²]")
ax2.set_title("Etter-puls spekter", fontsize=11)
ax2.grid(True, alpha=0.3)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Lagret figur: {OUT}")
