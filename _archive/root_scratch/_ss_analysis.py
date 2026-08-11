import pandas as pd
import numpy as np

df = pd.read_csv(r"casestudies/dyn_sim/results/WT1_LEOGO_FMU_results.csv")
t = df.t.values
a = df["fmu_YawBrTAyp"].values
t_end = t[-1]

# Verify the torque modulation was actually applied
if "torque_mod_factor" in df.columns:
    tm = df["torque_mod_factor"].values
    drv = t >= 15
    print(f"torque_mod_factor over t>=15s: min={tm[drv].min():.4f} "
          f"max={tm[drv].max():.4f}")
if "elecpwr_mod_factor" in df.columns:
    em = df["elecpwr_mod_factor"].values
    drv = t >= 15
    print(f"elecpwr_mod_factor over t>=15s: min={em[drv].min():.4f} "
          f"max={em[drv].max():.4f}")
if "fmu_GenTq" in df.columns:
    g = df["fmu_GenTq"].values
    drv = t >= 15
    print(f"GenTq over t>=15s: mean={g[drv].mean():.3f} "
          f"ripple(+/-)={np.ptp(g[drv])/2:.3f}")
if "fmu_BldPitch1" in df.columns:
    p = df["fmu_BldPitch1"].values
    drv = t >= 15
    print(f"BldPitch1 over t>=15s: mean={p[drv].mean():.4f} "
          f"ripple(+/-)={np.ptp(p[drv])/2:.5f}")

# Side-to-side natural frequency from zero-crossings (whole record)
m = (t >= 5) & (t <= t_end - 5)
tt, aa = t[m], a[m]
z = tt[np.where(np.diff(np.signbit(aa - aa.mean())))[0]]
per = 2 * np.mean(np.diff(z))
print(f"SS freq = {1/per:.4f} Hz (period {per:.3f} s)")

# Precise natural frequency from the PRE-forcing free decay (constant wind dir).
# The mode is very lightly damped, so this is the frequency to force at.
mpre = (t >= 20) & (t < 148)
if mpre.sum() > 100:
    tp, ap = t[mpre], a[mpre]
    zc = tp[np.where(np.diff(np.signbit(ap - ap.mean())))[0]]
    if len(zc) > 4:
        per_pre = 2 * np.mean(np.diff(zc))
        print(f"SS free-decay natural freq (t=20..148) = {1/per_pre:.5f} Hz "
              f"(period {per_pre:.4f} s, {len(zc)} crossings)")

# Amplitude envelope (half peak-to-peak) in 10 s windows every 25 s
print("amplitude envelope (half peak-to-peak):")
for c in np.arange(10, t_end - 4, 25):
    w = (t >= c - 5) & (t <= c + 5)
    if w.sum():
        print(f"  t={int(c):3d}s  amp={np.ptp(a[w]) / 2:.4f}")

# Steady-state forced amplitude: average envelope over the last 100 s
w_ss = t >= (t_end - 100)
ss_amp = np.ptp(a[w_ss]) / 2
print(f"\nSteady-state forced SS amplitude (last 100 s) = {ss_amp:.4f} m/s^2")

# Compare startup window (~30-40 s) vs steady state
w0 = (t >= 30) & (t <= 40)
print(f"Envelope at t~35 s (startup transient present) = {np.ptp(a[w0]) / 2:.4f} m/s^2")


def fft_peaks(t_arr, sig, f_lo=0.05, f_hi=6.0, n_peaks=6):
    """Return (freqs, mags) and print the strongest spectral peaks in a band."""
    dt = float(np.mean(np.diff(t_arr)))
    x = sig - np.mean(sig)
    # Hann window to reduce leakage
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(len(x), d=dt)
    mag = np.abs(X)
    band = (f >= f_lo) & (f <= f_hi)
    fb, mb = f[band], mag[band]
    # local maxima
    is_peak = np.r_[False, (mb[1:-1] > mb[:-2]) & (mb[1:-1] > mb[2:]), False]
    pk_idx = np.where(is_peak)[0]
    order = pk_idx[np.argsort(mb[pk_idx])[::-1]][:n_peaks]
    order = order[np.argsort(fb[order])]  # sort by frequency for readability
    for i in order:
        print(f"    peak at {fb[i]:.4f} Hz  (rel. magnitude {mb[i]/mb.max():.3f})")
    return fb, mb


# FFT to identify the tower side-to-side natural frequencies (mode 1 and mode 2).
# Use the pre-forcing window (constant wind direction) so both modes ring down
# freely and the higher (2nd) SS mode is not masked by the forced 1st-mode peak.
if np.any(t < 150):
    print("\nFFT of SS acceleration, PRE-forcing free decay (t = 5..150 s):")
    wpre = (t >= 5) & (t < 150)
    fft_peaks(t[wpre], a[wpre])

print("\nFFT of SS acceleration, FULL record:")
fft_peaks(t, a)

# ------------------------------------------------------------------
# CHIRP analysis: map the sweep time -> instantaneous forcing frequency
# and report where the SS-acceleration envelope peaks (= resonances).
# Must match the _make_winddir_hh.py --chirp parameters used for the run.
# ------------------------------------------------------------------
CHIRP_START, CHIRP_F0, CHIRP_F1, CHIRP_TEND = 50.0, 0.15, 2.5, 600.0
if abs(t_end - CHIRP_TEND) < 5:
    print(f"\nCHIRP envelope vs instantaneous forcing frequency "
          f"({CHIRP_F0}->{CHIRP_F1} Hz over t={CHIRP_START}..{CHIRP_TEND} s):")
    sweep = max(CHIRP_TEND - CHIRP_START, 1e-6)
    rows = []
    win_s = 3.0  # envelope window (s)
    for c in np.arange(CHIRP_START + win_s, CHIRP_TEND - win_s, win_s):
        w = (t >= c - win_s) & (t <= c + win_s)
        if w.sum() < 5:
            continue
        f_inst = CHIRP_F0 + (CHIRP_F1 - CHIRP_F0) * (c - CHIRP_START) / sweep
        env = np.ptp(a[w]) / 2
        rows.append((c, f_inst, env))
    rows = np.array(rows)
    env_max = rows[:, 2].max()
    for c, f_inst, env in rows:
        bar = "#" * int(round(40 * env / env_max))
        star = "  <== PEAK" if env > 0.6 * env_max else ""
        print(f"  t={c:6.1f}s  f={f_inst:5.3f} Hz  env={env:.4f}  {bar}{star}")
    # Local maxima in the envelope-vs-frequency curve
    e = rows[:, 2]
    is_pk = np.r_[False, (e[1:-1] > e[:-2]) & (e[1:-1] > e[2:]), False]
    pk = np.where(is_pk)[0]
    pk = pk[np.argsort(e[pk])[::-1]][:6]
    pk = pk[np.argsort(rows[pk, 1])]
    print("  --- resonance peaks (local maxima of envelope) ---")
    for i in pk:
        print(f"    resonance near f={rows[i,1]:.3f} Hz "
              f"(t={rows[i,0]:.1f}s, env={rows[i,2]:.4f}, "
              f"rel {rows[i,2]/env_max:.2f})")
