"""
Free-decay identification of the OpenFAST drivetrain torsional mode.

Purpose
-------
The drivetrain rotational-flexibility DOF (DrTrDOF) has been enabled in the
ElastoDyn model, with the torsional damper reduced to ~5% of critical so the
mode can ring. Before sweeping a torque disturbance around it, we need to know
its actual coupled natural frequency and damping. This script excites the
drivetrain with a brief generator-torque pulse and then lets it ring down, and
identifies the ring frequency (FFT) and damping ratio (log decrement) from the
high-speed-shaft torque HSShftTq.

Method
------
1. Write a ROSCO open-loop generator-torque table that holds the Region-2
   operating torque T0 constant everywhere, except for a short +/-kick over a
   narrow window [t_kick, t_kick + kick_dur]. Because the torque returns to T0,
   the operating point is unchanged and the torsional mode rings down freely.
2. Run the LEOGO + OpenFAST FMU co-simulation with no grid load event.
3. Analyse HSShftTq after the pulse: remove a slow trend (cubic polyfit), take
   the FFT to find the dominant ring frequency (masking < 0.2 Hz), and estimate
   the damping ratio from the decay of successive peak magnitudes.

Usage
-----
    python casestudies/dyn_sim/_decay_torsion.py            # run + analyse
    python casestudies/dyn_sim/_decay_torsion.py --plot-only
"""

from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# .../tops-openfast-sommer2026
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casestudies.dyn_sim.make_ol_gentq import write_ol_file  # noqa: E402

PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SIM = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_FMU_sim.py"
OL_FILE = PROJECT_ROOT / "test1002" / "ControlData" / "OLInput_ROSCO.dat"
DECAY_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "decay"

# Region-2 (8 m/s) operating-point generator torque [Nm].
T0_NM = 11023871.0


def write_pulse_ol_table(t_kick, kick_dur, kick_amp, t_max, dt=0.01):
    """Write a constant-T0 open-loop table with a brief torque kick.

    GenTq(t) = T0 * (1 + kick_amp)   for t_kick <= t < t_kick + kick_dur
             = T0                    otherwise
    """
    t = np.arange(0.0, t_max + dt, dt)
    factor = np.ones_like(t)
    in_kick = (t >= t_kick) & (t < t_kick + kick_dur)
    factor[in_kick] = 1.0 + kick_amp
    gentq = T0_NM * factor
    write_ol_file(t, gentq, OL_FILE)


def run_sim(t_end, out_csv):
    """Run one LEOGO + OpenFAST FMU co-simulation (no grid load event)."""
    cmd = [
        str(PY), str(SIM),
        "--t-end", str(t_end),
        "--load-step-mw", "0",
        "--out", str(out_csv),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def analyse(csv_path, t_lo, f_min=1.5):
    """Identify the drivetrain-torsion ring frequency and damping from HSShftTq.

    HSShftTq also carries slow tower/rotor content (< 1 Hz), so the torsion mode
    (~3 Hz) is isolated by masking the FFT below ``f_min`` and by band-passing the
    residual around the identified peak before the log-decrement fit.
    """
    d = pd.read_csv(csv_path)
    t = d["t"].values
    if "fmu_HSShftTq" in d.columns:
        y = d["fmu_HSShftTq"].values
        proxy = "fmu_HSShftTq"
    else:
        y = (d["fmu_GenSpeed"] - d["fmu_RotSpeed"]).values
        proxy = "fmu_GenSpeed - fmu_RotSpeed"

    m = t >= t_lo
    tt, yy = t[m], y[m]

    # Detrend: remove a slow (cubic) drift so the FFT/peaks see the ring only.
    coef = np.polyfit(tt, yy, 3)
    trend = np.polyval(coef, tt)
    r = yy - trend

    # FFT for the dominant ring frequency in the torsion band (mask below f_min).
    dt = np.median(np.diff(tt))
    n = len(r)
    win = np.hanning(n)
    Y = np.fft.rfft(r * win)
    f = np.fft.rfftfreq(n, dt)
    mag = np.abs(Y)
    band = f >= f_min
    f_peak = float(f[band][np.argmax(mag[band])])

    # Band-pass the residual around the torsion peak, then log-decrement on it.
    tors = _bandpass_fft(r, dt, 0.6 * f_peak, 1.6 * f_peak)
    zeta = _log_decrement_zeta(tt, tors, f_peak)

    return {
        "proxy": proxy,
        "f_peak_hz": f_peak,
        "zeta": zeta,
        "t": tt, "raw": yy, "trend": trend, "resid": r, "tors": tors,
        "f": f, "mag": mag,
    }


def _bandpass_fft(x, dt, f_lo, f_hi):
    """Zero-phase band-pass by masking the FFT to [f_lo, f_hi]."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, dt)
    X[(f < f_lo) | (f > f_hi)] = 0.0
    return np.fft.irfft(X, n=n)


def _log_decrement_zeta(t, r, f_peak, thresh_frac=0.1):
    """Estimate damping ratio from the decaying envelope of positive peaks.

    Only peaks above ``thresh_frac`` of the largest peak are used, so the flat
    numerical noise floor after ring-down does not flatten the fitted slope.
    The decay rate sigma [1/s] is fitted from ln(peak) vs peak time, and the
    damping ratio follows from sigma / (2*pi*f_peak).
    """
    # Local maxima of the band-passed signal.
    idx = np.where((r[1:-1] > r[:-2]) & (r[1:-1] > r[2:]))[0] + 1
    pk_t = t[idx]
    pk = r[idx]
    pos = pk > 0
    pk_t, pk = pk_t[pos], pk[pos]
    if len(pk) < 3:
        return float("nan")
    # Keep only the leading contiguous ring-down above the noise floor, so that
    # neither the flat tail nor the FFT edge artifact flattens the fitted slope.
    below = np.where(pk < thresh_frac * pk.max())[0]
    if len(below):
        cut = int(below[0])
        pk_t, pk = pk_t[:cut], pk[:cut]
    if len(pk) < 3:
        return float("nan")
    # sigma [1/s] from ln(peak) vs time.
    A = np.column_stack([pk_t, np.ones_like(pk_t)])
    slope, _ = np.linalg.lstsq(A, np.log(pk), rcond=None)[0]
    sigma = -slope
    omega = 2.0 * np.pi * f_peak
    zeta = sigma / np.sqrt(sigma**2 + omega**2)
    return float(zeta)


def plot(res, out_png, show=False):
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    ax[0].plot(res["t"], res["raw"], lw=0.8, label="HSShftTq")
    ax[0].plot(res["t"], res["trend"], "r--", lw=1.0, label="trend")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("HSShftTq [N m]")
    ax[0].set_title(
        f"Drivetrain torsion free decay: "
        f"f = {res['f_peak_hz']:.3f} Hz, zeta = {100*res['zeta']:.2f} %"
    )
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(res["t"], res["tors"], lw=0.8, color="tab:green")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("torsion band-pass [N m]")
    ax[1].set_title(
        f"Band-passed around {res['f_peak_hz']:.3f} Hz (torsion ring-down)"
    )
    ax[1].grid(True, alpha=0.3)

    band = res["f"] <= 8.0
    ax[2].plot(res["f"][band], res["mag"][band], lw=1.0)
    ax[2].axvline(res["f_peak_hz"], color="r", ls="--",
                  label=f"{res['f_peak_hz']:.3f} Hz")
    ax[2].set_xlabel("frequency [Hz]")
    ax[2].set_ylabel("|FFT| of detrended HSShftTq")
    ax[2].legend(loc="best")
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t-end", type=float, default=40.0)
    p.add_argument("--t-kick", type=float, default=10.0)
    p.add_argument("--kick-dur", type=float, default=0.3,
                   help="Duration of the torque kick [s].")
    p.add_argument("--kick-amp", type=float, default=0.10,
                   help="Fractional torque kick amplitude (0.10 = +10%%).")
    p.add_argument("--t-analyse-from", type=float, default=11.0,
                   help="Start of the ring-down analysis window [s].")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    csv_path = DECAY_DIR / "torsion_decay.csv"
    png_path = DECAY_DIR / "torsion_decay.png"

    if not args.plot_only:
        write_pulse_ol_table(args.t_kick, args.kick_dur, args.kick_amp,
                             t_max=args.t_end)
        run_sim(args.t_end, csv_path)

    res = analyse(csv_path, args.t_analyse_from)
    print(f"proxy          = {res['proxy']}")
    print(f"ring frequency = {res['f_peak_hz']:.4f} Hz")
    print(f"damping ratio  = {100*res['zeta']:.3f} %")
    plot(res, png_path, show=args.show)


if __name__ == "__main__":
    main()
