"""
Tower side-to-side (SS) resonance sweep driven by a *genuine grid disturbance*.

Unlike sweep_ss_reduced.py (which modulates the generator torque Te directly)
and sweep_resonance.py (which imposes the torque via the ROSCO open-loop table),
this sweep excites the turbine only through the electrical network: a sustained
sinusoidal real-power load modulation is applied at the main LEOGO grid bus and
its frequency is swept. The disturbance reaches the tower purely through the
open electrical -> mechanical path

    grid load modulation -> bus V/f oscillation -> UIC power Pe
        -> generator torque Te -> nacelle reaction moment -> tower SS mode.

For each drive frequency the steady-state SS acceleration amplitude is estimated
by a single-bin lock-in (Fourier projection) over the settled window. A load
step (or a fault) would be broadband and could not build the mode up; a sustained
sinusoid at ~f_ss can. The curve should therefore peak sharply at the tower SS
natural frequency (~0.234 Hz), close to the LEOGO grid electromechanical/COI
mode (~0.226 Hz).

Run from anywhere; it shells out to test_WT_LEOGO_tower_sim.py per frequency.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVER = PROJECT_ROOT / 'casestudies' / 'dyn_sim' / 'test_WT_LEOGO_tower_sim.py'
PYTHON = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
OUT_DIR = PROJECT_ROOT / 'results' / 'sweep_ss_grid'

# --- Sweep configuration ---
GRID_MOD_AMP_MW = 5.0        # sinusoidal load-modulation amplitude [MW]
GRID_MOD_START = 10.0        # modulation onset [s]
T_END = 70.0                 # simulation length [s]
MEASURE_START = 45.0         # steady-state window start [s] (~2.5 tau after onset)
F_SS = 0.234                 # tower SS natural frequency [Hz]
F_GRID_COI = 0.226           # LEOGO grid electromechanical/COI mode [Hz]

FREQS_HZ = np.array([
    0.16, 0.18, 0.20, 0.21, 0.220, 0.226, 0.230, 0.234,
    0.238, 0.245, 0.26, 0.28, 0.30,
])


def lockin_amplitude(t, x, f, t0):
    """Amplitude of the component of x(t) at frequency f over t >= t0.

    Single-bin Fourier projection: A = 2/N * |sum x * exp(-j 2 pi f t)|.
    Robust to the small remaining transient and to DC offset (mean removed).
    """
    m = t >= t0
    tw = t[m]
    xw = x[m] - np.mean(x[m])
    ph = np.exp(-1j * 2.0 * np.pi * f * tw)
    return 2.0 * np.abs(np.sum(xw * ph)) / len(xw)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / 'runs'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, f in enumerate(FREQS_HZ):
        csv_rel = f'results/sweep_ss_grid/runs/ss_grid_{f:.3f}.csv'
        csv_abs = PROJECT_ROOT / csv_rel
        cmd = [
            str(PYTHON), str(DRIVER),
            '--grid-mod-amp', str(GRID_MOD_AMP_MW),
            '--grid-mod-freq', f'{f:.4f}',
            '--grid-mod-start', str(GRID_MOD_START),
            '--t-end', str(T_END),
            '--out', csv_rel,
        ]
        print(f'[{i+1}/{len(FREQS_HZ)}] f = {f:.3f} Hz ...', flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        df = pd.read_csv(csv_abs)
        t = df['t'].to_numpy()
        a = df['ss_accel_mps2'].to_numpy()
        amp = lockin_amplitude(t, a, f, MEASURE_START)
        m = t >= MEASURE_START
        std = float(np.std(a[m]))
        p2p = float(np.max(a[m]) - np.min(a[m]))
        records.append({
            'freq_hz': float(f),
            'ss_amp_lockin_mps2': float(amp),
            'ss_std_mps2': std,
            'ss_p2p_mps2': p2p,
        })
        print(f'      lock-in amp = {amp:.4f} m/s^2 (std {std:.4f}, p2p {p2p:.4f})',
              flush=True)

    summary = pd.DataFrame(records)
    summary_csv = OUT_DIR / 'ss_grid_summary.csv'
    summary.to_csv(summary_csv, index=False)
    print(f'\nSummary written: {summary_csv}')

    f = summary['freq_hz'].to_numpy()
    amp = summary['ss_amp_lockin_mps2'].to_numpy()
    f_peak = f[np.argmax(amp)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f, amp, 'o-', lw=1.6, label='SS accel amplitude (lock-in)')
    ax.axvline(F_SS, color='g', ls='--', lw=1.0, label=f'tower SS {F_SS:.3f} Hz')
    ax.axvline(F_GRID_COI, color='orange', ls=':', lw=1.2,
               label=f'grid COI {F_GRID_COI:.3f} Hz')
    ax.axvline(f_peak, color='m', ls='-.', lw=0.9, label=f'peak {f_peak:.3f} Hz')
    ax.set_xlabel('Grid load-modulation frequency [Hz]')
    ax.set_ylabel('Tower SS acceleration amplitude [m/s$^2$]')
    ax.set_title(f'SS resonance via genuine grid disturbance '
                 f'(+/-{GRID_MOD_AMP_MW:.0f} MW load modulation)')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    png = OUT_DIR / 'ss_grid_curve.png'
    fig.savefig(png, dpi=130)
    print(f'Figure written: {png}')
    print(f'Peak SS response at {f_peak:.3f} Hz '
          f'(tower SS mode = {F_SS:.3f} Hz).')


if __name__ == '__main__':
    main()
