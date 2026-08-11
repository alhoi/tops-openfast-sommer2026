"""
Frequency sweep of the OpenFAST drivetrain torsional mode via the ROSCO
open-loop generator-torque channel.

This is the torsional counterpart to ``sweep_resonance.py`` (which sweeps the
tower side-to-side mode). The free-decay identification (``_decay_torsion.py``)
placed the drivetrain torsion mode at ~3.10 Hz with ~5% damping. Here we probe
that mode by sweeping the generator-torque modulation frequency and measuring
how strongly the high-speed-shaft torque HSShftTq responds at each frequency.
A resonance peak at ~3.10 Hz demonstrates the frequency-selective coupling of
the electrical torque channel to the mechanical torsion mode.

Torque path
-----------
The modulation is prescribed as a sinusoidal generator torque

    GenTq(t) = T0 * (1 + amp * sin(2*pi*f*(t - t_start)))   for t >= t_start

through the ROSCO open-loop table (OLInput_ROSCO.dat, OL_Mode=1) -- the only
torque channel that reaches the OpenFAST structure. The torque source is
imposed, but the frequency selectivity of the torsional response is genuine
OpenFAST structural physics.

Measurement
-----------
For each frequency we run the LEOGO + OpenFAST FMU co-simulation with no grid
load event, then fit a sinusoid at the probe frequency to HSShftTq over a
settled window. The fitted amplitude isolates the response at the forcing
frequency from the mean shaft torque and broadband motion. Because the imposed
torque ripple is also transmitted directly through the shaft, the curve sits on
a non-zero baseline; the resonant amplification appears as a peak on top of it.

Usage
-----
    python casestudies/dyn_sim/sweep_torsion.py                # run + plot
    python casestudies/dyn_sim/sweep_torsion.py --plot-only    # re-plot only
    python casestudies/dyn_sim/sweep_torsion.py --skip-existing # resume
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
SWEEP_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "sweep_torsion"

# Region-2 (8 m/s) operating-point generator torque [Nm].
T0_NM = 11023871.0

# Drivetrain torsion mode from the free-decay identification.
F_TORSION = 3.10

# Measured channel: high-speed-shaft torque.
CHANNEL = "fmu_HSShftTq"


def write_sine_ol_table(freq_hz, amp, t_start, t_max, dt=0.01):
    """Write a sinusoidal generator-torque open-loop table for one frequency."""
    t = np.arange(0.0, t_max + dt, dt)
    factor = np.where(
        t >= t_start,
        1.0 + amp * np.sin(2.0 * np.pi * freq_hz * (t - t_start)),
        1.0,
    )
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


def fitted_amplitude(csv_path, freq_hz, t_lo, t_hi):
    """Least-squares amplitude of HSShftTq at freq_hz over [t_lo, t_hi].

    Fits  y ~ a*sin(w t) + b*cos(w t) + c  and returns sqrt(a^2 + b^2),
    isolating the response at the forcing frequency from the mean shaft torque.
    Also returns the raw std and p2p.
    """
    d = pd.read_csv(csv_path)
    t = d["t"].values
    y = d[CHANNEL].values
    m = (t >= t_lo) & (t <= t_hi)
    tt, yy = t[m], y[m]

    w = 2.0 * np.pi * freq_hz
    A = np.column_stack([np.sin(w * tt), np.cos(w * tt), np.ones_like(tt)])
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    a, b, _c = coef
    amp = float(np.hypot(a, b))
    return amp, float(yy.std()), float(yy.max() - yy.min())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--freqs", type=float, nargs="+",
        default=[2.6, 2.8, 2.9, 3.0, 3.05, 3.10, 3.15, 3.2, 3.3, 3.5, 3.8],
        help="Probe frequencies [Hz].",
    )
    p.add_argument("--amp", type=float, default=0.10,
                   help="Fractional torque-modulation amplitude (0.10 = +/-10%%).")
    p.add_argument("--t-end", type=float, default=30.0,
                   help="Simulation end time [s].")
    p.add_argument("--t-start", type=float, default=10.0,
                   help="Time [s] at which the modulation switches on.")
    p.add_argument("--win", type=float, default=15.0,
                   help="Length [s] of the settled measurement window ending at t-end.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse per-frequency CSVs that already exist (resume a sweep).")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip simulations; rebuild the curve from existing CSVs.")
    p.add_argument("--show", action="store_true", help="Show the figure interactively.")
    args = p.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    t_lo = args.t_end - args.win
    t_hi = args.t_end

    records = []
    for i, f in enumerate(args.freqs, 1):
        tag = f"{f:.3f}".replace(".", "p")
        out_csv = SWEEP_DIR / f"tors_{tag}.csv"

        if not args.plot_only:
            if args.skip_existing and out_csv.is_file():
                print(f"[{i}/{len(args.freqs)}] f={f:.3f} Hz -> reuse {out_csv.name}")
            else:
                print(f"[{i}/{len(args.freqs)}] f={f:.3f} Hz -> "
                      f"writing OL table + running sim ...", flush=True)
                write_sine_ol_table(f, args.amp, args.t_start, args.t_end)
                run_sim(args.t_end, out_csv)

        if not out_csv.is_file():
            print(f"  WARNING: {out_csv} missing, skipping this point.")
            continue

        amp, std, p2p = fitted_amplitude(out_csv, f, t_lo, t_hi)
        records.append({"freq_hz": f, "tors_amp_fit": amp,
                        "tors_std": std, "tors_p2p": p2p})
        print(f"    f={f:.3f} Hz: HSShftTq fitted amp={amp:.4e}  "
              f"std={std:.4e}  p2p={p2p:.4e}")

    if not records:
        print("No results to plot.")
        return

    summary = pd.DataFrame(records).sort_values("freq_hz").reset_index(drop=True)
    summary_csv = SWEEP_DIR / "torsion_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nSummary written to: {summary_csv}")

    # ---- resonance curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(summary["freq_hz"], summary["tors_amp_fit"] / 1e6,
            "o-", color="tab:blue", label="Drivetrain torsion response (fitted amp.)")
    ax.axvline(F_TORSION, color="tab:red", ls="--", lw=1.2,
               label=f"Torsion mode {F_TORSION:.2f} Hz")
    ax.set_xlabel("Torque-modulation frequency [Hz]")
    ax.set_ylabel(r"HSShftTq amplitude at forcing freq. [MN$\cdot$m]")
    ax.set_title(
        f"Electro-mechanical torsion resonance sweep "
        f"(\u00b1{args.amp*100:.0f}% gen. torque, Region 2)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    fig_png = SWEEP_DIR / "torsion_curve.png"
    fig.savefig(fig_png, dpi=150)
    print(f"Figure written to: {fig_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
