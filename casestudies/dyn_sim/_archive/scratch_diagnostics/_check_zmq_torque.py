"""Quick check: does the ZMQ torque offset reach the generator/structure?
Fits sinusoids at f=0.234 Hz to (a) the server-side offset+GenTqMeas log and
(b) the co-sim CSV fmu_GenTq / YawBrTAyp. Prints amplitudes.
"""
import sys
import numpy as np
import pandas as pd

F = 0.234


def sine_fit(t, y, f):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y = y - y.mean()
    w = 2 * np.pi * f
    A = np.c_[np.cos(w * t), np.sin(w * t)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.hypot(c[0], c[1]))


def col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} in {list(df.columns)[:12]}...")


srv = r"results\sweep\zmq_server_log.csv"
sim = r"results\sweep\diag_zmq_torque.csv"
onset = 15.0

print("=== SERVER LOG ===")
try:
    s = pd.read_csv(srv)
    print("cols:", list(s.columns))
    tc = col(s, "t", "time", "Time")
    m = s[tc] >= onset
    for c in s.columns:
        if c == tc:
            continue
        try:
            amp = sine_fit(s[tc][m], s[c][m], F)
            print(f"  {c:16s} mean={s[c][m].mean():.4e}  amp@{F}Hz={amp:.4e}")
        except Exception as e:
            print(f"  {c}: {e}")
except Exception as e:
    print("server log error:", e)

print("\n=== CO-SIM CSV ===")
d = pd.read_csv(sim)
tc = col(d, "t", "time", "Time")
m = d[tc] >= onset
gt = col(d, "fmu_GenTq", "GenTq")
ss = col(d, "fmu_YawBrTAyp", "YawBrTAyp")
print(f"  window t>= {onset}s, n={m.sum()}")
print(f"  {gt} mean={d[gt][m].mean():.2f}  amp@{F}Hz={sine_fit(d[tc][m], d[gt][m], F):.2f}  std={d[gt][m].std():.2f}  ptp={np.ptp(d[gt][m]):.2f}")
print(f"  {ss} mean={d[ss][m].mean():.4f}  amp@{F}Hz={sine_fit(d[tc][m], d[ss][m], F):.4f}  std={d[ss][m].std():.4f}")
# also GenSpeed / RotSpeed if present
for name in ("fmu_GenSpeed", "fmu_RotSpeed", "fmu_HSShftTq"):
    if name in d.columns:
        print(f"  {name} mean={d[name][m].mean():.4f}  amp@{F}Hz={sine_fit(d[tc][m], d[name][m], F):.4e}")
