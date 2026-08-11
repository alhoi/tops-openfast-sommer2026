"""Characterise the high-frequency 'thickness' of the red (ON) curves vs OFF.
Detrends (moving average) each signal in a settled window, reports std and the
dominant FFT frequency/amplitude of the fast component."""
import numpy as np
import pandas as pd

WIN = (70.0, 120.0)
files = [("OFF", r"results\sweep\r3_gt_off.csv"),
         ("ON ", r"results\sweep\r3_gt_on3.csv")]


def hf(t, y, smooth_s=1.0):
    dt = np.median(np.diff(t))
    n = max(1, int(smooth_s / dt))
    trend = pd.Series(y).rolling(n, center=True, min_periods=1).mean().values
    r = y - trend
    # FFT of the residual
    r = r - r.mean()
    Y = np.abs(np.fft.rfft(r * np.hanning(len(r))))
    f = np.fft.rfftfreq(len(r), dt)
    k = np.argmax(Y[1:]) + 1
    amp = 2 * Y[k] / np.sum(np.hanning(len(r)))
    return float(np.std(r)), float(f[k]), float(amp)


for tag, path in files:
    d = pd.read_csv(path)
    m = (d["t"] >= WIN[0]) & (d["t"] <= WIN[1])
    t = d["t"][m].values
    print(f"--- {tag}  window {WIN} ---")
    for col, unit in [("fmu_GenTq", "kNm"), ("f_grid_hz", "Hz"),
                      ("fmu_YawBrTAyp", "m/s2"), ("P_uic_bus_sys_pu", "pu"),
                      ("zmq_torque_offset_nm", "Nm")]:
        if col in d.columns:
            s, fpk, amp = hf(t, d[col][m].values)
            print(f"  {col:22s} HF std={s:.4e}  dom {fpk:5.2f} Hz  amp={amp:.4e} {unit}")
