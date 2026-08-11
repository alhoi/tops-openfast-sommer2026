"""Identify the blade edgewise natural frequency from a torque-kick run.

Blade edgewise is a rotating-frame in-plane mode; the FMU does not expose blade
deflection, so it is detected in the shaft torque (HSShftTq) and rotor speed
(RotSpeed): the collective edgewise mode directly modulates the shaft torque.
FFTs both channels post-pulse and reports the spectral peaks, with the rotor
harmonics (nP) drawn for reference so the structural edgewise peak stands out.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV = (Path(sys.argv[1]) if len(sys.argv) > 1
       else PROJECT_ROOT / "results" / "sweep" / "edge_freq_id.csv")
OUT = (PROJECT_ROOT / "results" / "em_interaction_sweep"
       / (CSV.stem + "_spectrum.png"))

PULSE_END = 11.0
BAND = (0.15, 3.0)
CHANNELS = [("fmu_HSShftTq", "Akselmoment\nHSShftTq [kNm]"),
            ("fmu_RotSpeed", "Rotorhastighet\nRotSpeed [rpm]")]


def _spectrum(t, y):
    dt = float(np.median(np.diff(t)))
    win = np.hanning(len(y))
    spec = np.abs(np.fft.rfft((y - np.mean(y)) * win)) * (2.0 / np.sum(win))
    freqs = np.fft.rfftfreq(len(y), dt)
    return freqs, spec


def _peaks(freqs, spec):
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    fb, sb = freqs[band], spec[band]
    idx, _ = find_peaks(sb, height=0.08 * sb.max())
    idx = idx[np.argsort(sb[idx])[::-1]][:5]
    return sorted(float(f) for f in fb[idx])


df = pd.read_csv(CSV)
t = df["t"].to_numpy()
m = t >= PULSE_END
tt = t[m]

rot_rpm = float(np.median(df["fmu_RotSpeed"].to_numpy()[m]))
onep = rot_rpm / 60.0
print(f"Rotor ~{rot_rpm:.2f} rpm -> 1P={onep:.3f} Hz, 3P={3 * onep:.3f} Hz, "
      f"6P={6 * onep:.3f} Hz")

fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(9, 3.0 * len(CHANNELS)))
for ax, (ch, label) in zip(axes, CHANNELS):
    y = df[ch].to_numpy()[m]
    freqs, spec = _spectrum(tt, y)
    pk = _peaks(freqs, spec)
    print(f"{ch} peaks in {BAND[0]}-{BAND[1]} Hz: "
          + ", ".join(f"{p:.3f}" for p in pk) + " Hz")
    ax.plot(freqs, spec, color="#0b6e4f", lw=1.2)
    for p in pk:
        s_at = spec[np.argmin(np.abs(freqs - p))]
        ax.axvline(p, color="#c0392b", ls="--", lw=0.8, alpha=0.7)
        ax.annotate(f"{p:.3f}", xy=(p, s_at), fontsize=8, color="#c0392b")
    for n in (1, 3, 6):
        ax.axvline(n * onep, color="#888", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("Frekvens [Hz]")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
axes[0].set_title("Blad kantvis: etter-puls spekter (nP-harmoniske stiplet grått)",
                  fontsize=11)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Lagret figur: {OUT}")
