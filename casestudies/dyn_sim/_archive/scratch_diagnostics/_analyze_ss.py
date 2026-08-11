import pandas as pd
import numpy as np

d = pd.read_csv('casestudies/dyn_sim/results/WT1_LEOGO_FMU_results.csv')
print('COLS:', list(d.columns))

# locate time column
tcol = 'time' if 'time' in d.columns else d.columns[0]
t = d[tcol].values


def pick(*names):
    for n in names:
        for c in d.columns:
            if n.lower() in c.lower():
                return c
    return None

gen = pick('GenTq')
rot = pick('RotSpeed')
ss = pick('YawBrTAyp')
fgrid = pick('f_grid_hz')
print('gen=', gen, 'rot=', rot, 'ss=', ss, 'fgrid=', fgrid)
print('t range', t.min(), t.max(), 'N', len(t))


def win(a, b):
    m = (t >= a) & (t < b)
    return m

# Grid frequency event (drives the droop torque)
if fgrid:
    f = d[fgrid].values
    f0 = f[win(0, 5)].mean()
    print(f'f_grid pre-event={f0:.4f} Hz  min={f.min():.4f}  max={f.max():.4f}  '
          f'dip={f.min()-f0:+.4f} Hz at t={t[int(np.argmin(f))]:.2f}s')

# GenTq tracking check over whole modulation window
if gen:
    g = d[gen].values
    m = win(10, t.max())
    print(f'GenTq (OL droop) t>10: mean={g[m].mean():.1f} min={g[m].min():.1f} max={g[m].max():.1f} p2p={g[m].max()-g[m].min():.1f} kNm')

if rot:
    r = d[rot].values
    print(f'RotSpeed start={r[win(0,5)].mean():.4f} end={r[win(t.max()-10,t.max())].mean():.4f} rpm')

if ss:
    s = d[ss].values
    windows = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 55), (55, 70), (70, 85), (85, 100)]
    print('\nYawBrTAyp (tower side-to-side) windowed std / p2p:')
    for a, b in windows:
        m = win(a, b)
        if m.sum() > 5:
            sub = s[m]
            print(f'  t={a:3d}-{b:3d}s: std={sub.std():.4f}  p2p={sub.max()-sub.min():.4f}  n={m.sum()}')
    i0 = np.searchsorted(t, 10)
    ipk = i0 + int(np.argmax(np.abs(s[i0:])))
    print(f'\n  |YawBrTAyp| peak after event = {abs(s[ipk]):.4f} at t={t[ipk]:.2f}s')
