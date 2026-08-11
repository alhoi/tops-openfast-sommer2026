"""
Frequency sweep of the reduced-model tower side-to-side (SS) mode.

This is the reduced-model counterpart of sweep_resonance.py (which sweeps the
OpenFAST-FMU SS mode). It probes the WindTurbineTower SS modal oscillator by
modulating the generator electromagnetic torque that drives it,

    Te_drive(t) = Te * (1 + amp * sin(2*pi*f*(t - t_start))),

and measuring the tower-top side-to-side acceleration (ss_accel_mps2) response
at each frequency. The modulation is injected through the model's
set_ss_test_modulation hook via test_WT_LEOGO_tower_sim.py --ss-mod-*.

Purpose
-------
1. Verify the implemented SS oscillator reproduces a clean second-order
   resonance at f_ss (~0.234 Hz).
2. Calibrate the coupling gain g_ss against the FMU sweep: the FMU measured a
   peak SS acceleration of ~0.259 m/s^2 for +/-10 % generator-torque modulation.
   Because the oscillator is linear in g_ss, the calibrated gain is

       g_ss_new = g_ss_used * (A_peak_FMU / A_peak_reduced).

Note on operating point: the reduced model runs at its hard-coded 11 m/s
(Region 3), whereas the FMU sweep was at 8 m/s (Region 2). The peak-amplitude
match (fractional-modulation basis) folds that difference into g_ss.

Usage
-----
    python casestudies/dyn_sim/sweep_ss_reduced.py               # run + plot
    python casestudies/dyn_sim/sweep_ss_reduced.py --plot-only   # re-plot only
    python casestudies/dyn_sim/sweep_ss_reduced.py --skip-existing
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

PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SIM = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_tower_sim.py"
SWEEP_DIR = PROJECT_ROOT / "results" / "sweep_ss_reduced"

# FMU sweep reference (from sweep_resonance.py results) for calibration.
A_PEAK_FMU = 0.259   # m/s^2, peak YawBrTAyp at ~0.234 Hz for +/-10% torque
F_TOWER_SS = 0.234   # tower side-to-side 1st mode
F_GRID_COI = 0.226   # LEOGO grid centre-of-inertia electromechanical mode


def run_sim(freq_hz, amp, t_start, t_end, g_ss, out_csv):
    """Run one reduced-model LEOGO + WindTurbineTower simulation."""
    cmd = [
        str(PY), str(SIM),
        "--t-end", str(t_end),
        "--load-step-mw", "0",
        "--g-ss", str(g_ss),
        "--ss-mod-amp", str(amp),
        "--ss-mod-freq", str(freq_hz),
        "--ss-mod-start", str(t_start),
        "--out", str(out_csv),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def fitted_amplitude(csv_path, freq_hz, t_lo, t_hi):
    """Least-squares amplitude of ss_accel_mps2 at freq_hz over [t_lo, t_hi]."""
    d = pd.read_csv(csv_path)
    t = d["t"].values
    y = d["ss_accel_mps2"].values
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
        default=[0.10, 0.19, 0.21, 0.226, 0.234, 0.245, 0.26, 0.30, 0.40],
        help="Probe frequencies [Hz].",
    )
    p.add_argument("--amp", type=float, default=0.10,
                   help="Fractional torque-modulation amplitude (0.10 = +/-10%%).")
    p.add_argument("--g-ss", type=float, default=0.259,
                   help="SS coupling gain used for the sweep runs.")
    p.add_argument("--t-end", type=float, default=90.0,
                   help="Simulation end time [s] (>= ~6/(zeta*f_ss) to settle).")
    p.add_argument("--t-start", type=float, default=10.0,
                   help="Time [s] at which the modulation switches on.")
    p.add_argument("--win", type=float, default=30.0,
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
                print(f"[{i}/{len(args.freqs)}] f={f:.3f} Hz -> running sim ...",
                      flush=True)
                run_sim(f, args.amp, args.t_start, args.t_end, args.g_ss, out_csv)

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
    summary_csv = SWEEP_DIR / "ss_reduced_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nSummary written to: {summary_csv}")

    # ---- calibration report --------------------------------------------
    i_peak = int(summary["ss_amp_fit"].values.argmax())
    f_peak = float(summary["freq_hz"].values[i_peak])
    a_peak = float(summary["ss_amp_fit"].values[i_peak])
    g_ss_new = args.g_ss * (A_PEAK_FMU / a_peak) if a_peak > 0 else float("nan")
    print("\n--- Calibration ---")
    print(f"Reduced-model peak: {a_peak:.4f} m/s^2 at {f_peak:.3f} Hz "
          f"(g_ss used = {args.g_ss:.4f})")
    print(f"FMU peak target:    {A_PEAK_FMU:.4f} m/s^2")
    print(f"=> calibrated g_ss = {g_ss_new:.4f} "
          f"(scale by {A_PEAK_FMU / a_peak:.4f})")

    # ---- resonance curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(summary["freq_hz"], summary["ss_amp_fit"],
            "o-", color="tab:blue", label="Reduced-model SS response (fitted amp.)")
    ax.axhline(A_PEAK_FMU, color="tab:orange", ls="-.", lw=1.2,
               label=f"FMU peak {A_PEAK_FMU:.3f} m/s$^2$")
    ax.axvline(F_TOWER_SS, color="tab:red", ls="--", lw=1.2,
               label=f"Tower SS mode {F_TOWER_SS:.3f} Hz")
    ax.axvline(F_GRID_COI, color="tab:green", ls=":", lw=1.2,
               label=f"LEOGO grid mode {F_GRID_COI:.3f} Hz")
    ax.set_xlabel("Torque-modulation frequency [Hz]")
    ax.set_ylabel(r"SS acceleration amplitude at forcing freq. [m/s$^2$]")
    ax.set_title(
        f"Reduced-model SS resonance sweep "
        f"(\u00b1{args.amp*100:.0f}% gen. torque, g_ss={args.g_ss:.3f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    fig_png = SWEEP_DIR / "ss_reduced_curve.png"
    fig.savefig(fig_png, dpi=150)
    print(f"Figure written to: {fig_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
