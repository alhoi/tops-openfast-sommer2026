"""
Frequency sweep of the OpenFAST tower side-to-side (SS) mode via the ROSCO
open-loop generator-torque channel.

Motivation
----------
The research goal is to demonstrate electro-mechanical interaction: a grid
disturbance in the LEOGO platform network exciting a lightly damped mechanical
mode of the offshore wind turbine. The tower side-to-side (SS) 1st mode sits at
~0.234 Hz, and the LEOGO grid centre-of-inertia electromechanical mode sits at
~0.226 Hz -- essentially on top of it. So a grid oscillation in that band is
exactly where the coupling is physically strongest.

This script probes that coupling by sweeping the generator-torque modulation
frequency and measuring how strongly the tower SS mode responds at each
frequency. A resonance peak at ~0.234 Hz demonstrates the frequency-selective
electro-mechanical coupling.

Torque path
-----------
The modulation is prescribed as a sinusoidal generator torque

    GenTq(t) = T0 * (1 + amp * sin(2*pi*f*(t - t_start)))   for t >= t_start

through the ROSCO open-loop table (OLInput_ROSCO.dat, OL_Mode=1). This is the
only torque channel that actually reaches the OpenFAST structure: ROSCO
(VSContrl=5) overrides the external GenSpdOrTrq / ElecPwrCom inputs, so the
open-loop table is the physically effective injection point. The torque source
is therefore imposed, but the frequency selectivity of the tower response is
genuine OpenFAST structural physics.

Measurement
-----------
For each frequency we run the LEOGO + OpenFAST FMU co-simulation with no grid
load event (a clean single-frequency probe), then fit a sinusoid at the probe
frequency to the tower-top side-to-side acceleration (YawBrTAyp) over a settled
window. The fitted amplitude isolates the response at the forcing frequency
from broadband ambient motion, giving a clean resonance curve.

Usage
-----
    python casestudies/dyn_sim/sweep_resonance.py                # run + plot
    python casestudies/dyn_sim/sweep_resonance.py --plot-only    # re-plot only
    python casestudies/dyn_sim/sweep_resonance.py --skip-existing # resume
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
SWEEP_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "sweep"

# Region-2 (8 m/s) operating-point generator torque [Nm].
T0_NM = 11023871.0

# Reference mode frequencies for annotation.
F_TOWER_SS = 0.234   # OpenFAST tower side-to-side 1st mode
F_GRID_COI = 0.226   # LEOGO grid centre-of-inertia electromechanical mode


def write_sine_ol_table(freq_hz, amp, t_start, t_max, dt=0.05):
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
    """Least-squares amplitude of YawBrTAyp at freq_hz over [t_lo, t_hi].

    Fits  y ~ a*sin(w t) + b*cos(w t) + c  and returns sqrt(a^2 + b^2),
    which isolates the response at the forcing frequency from the DC offset
    and broadband ambient motion. Also returns the raw std and p2p.
    """
    d = pd.read_csv(csv_path)
    t = d["t"].values
    y = d["fmu_YawBrTAyp"].values
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
        default=[0.10, 0.15, 0.19, 0.21, 0.226, 0.234, 0.245, 0.26, 0.30, 0.40],
        help="Probe frequencies [Hz].",
    )
    p.add_argument("--amp", type=float, default=0.10,
                   help="Fractional torque-modulation amplitude (0.10 = +/-10%%).")
    p.add_argument("--t-end", type=float, default=60.0,
                   help="Simulation end time [s].")
    p.add_argument("--t-start", type=float, default=10.0,
                   help="Time [s] at which the modulation switches on.")
    p.add_argument("--win", type=float, default=35.0,
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
        out_csv = SWEEP_DIR / f"ss_{tag}.csv"

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
        records.append({"freq_hz": f, "ss_amp_fit": amp,
                        "ss_std": std, "ss_p2p": p2p})
        print(f"    f={f:.3f} Hz: SS fitted amp={amp:.4f}  std={std:.4f}  p2p={p2p:.4f}")

    if not records:
        print("No results to plot.")
        return

    summary = pd.DataFrame(records).sort_values("freq_hz").reset_index(drop=True)
    summary_csv = SWEEP_DIR / "resonance_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nSummary written to: {summary_csv}")

    # ---- resonance curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(summary["freq_hz"], summary["ss_amp_fit"],
            "o-", color="tab:blue", label="Tower SS response (fitted amp.)")
    ax.axvline(F_TOWER_SS, color="tab:red", ls="--", lw=1.2,
               label=f"Tower SS mode {F_TOWER_SS:.3f} Hz")
    ax.axvline(F_GRID_COI, color="tab:green", ls=":", lw=1.2,
               label=f"LEOGO grid mode {F_GRID_COI:.3f} Hz")
    ax.set_xlabel("Torque-modulation frequency [Hz]")
    ax.set_ylabel(r"YawBrTAyp amplitude at forcing freq. [m/s$^2$]")
    ax.set_title(
        f"Electro-mechanical resonance sweep "
        f"(\u00b1{args.amp*100:.0f}% gen. torque, Region 2)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    fig_png = SWEEP_DIR / "resonance_curve.png"
    fig.savefig(fig_png, dpi=150)
    print(f"Figure written to: {fig_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
