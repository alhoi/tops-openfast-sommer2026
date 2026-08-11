"""
Generate the 15-family deep droop diagnostics (no-droop vs droop) for the
three electromechanical scenarios we analyse.

For every scenario this script:
  1. runs test_WT_LEOGO_droop_comparison_diagnostics.py (plain WindTurbine,
     full pitch / torque / converter logging) which writes the two canonical
     rich CSV files (NOdroop + droop),
  2. runs plot_WT_LEOGO_diagnostics_comparison.py to build the 15 comparison
     plots into a per-scenario folder,
  3. copies the two CSV files next to the plots so the data is preserved
     before the next scenario overwrites the canonical files.

All scenarios use the same droop settings as the mode figures:
K_droop = 0.75 pu/Hz, headroom = 0.05 pu (1 MW on the 20 MVA UIC base).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

DIAG = "casestudies/dyn_sim/test_WT_LEOGO_droop_comparison_diagnostics.py"
PLOT = "casestudies/dyn_sim/plotting/plot_WT_LEOGO_diagnostics_comparison.py"

CSV_DIR = ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
DIAG_PLOTS = ROOT / "results" / "em_interaction_sweep"

CSV_NAMES = (
    "WT1_LEOGO_frequency_Sine_freq_NOdroop.csv",
    "WT1_LEOGO_frequency_Sine_freq_droop.csv",
)

# Matched droop settings (same as the mode figures).
DROOP = ["--droop-gain-pu-per-hz", "0.75", "--headroom-pu", "0.05"]

SCENARIOS = [
    {
        # Hero-like broadband 10 MW load step: mean=1, amplitude=0 -> a step
        # that stays on until t_end (permanent, like the headline figure).
        "name": "Hero_10MW_step",
        "sim": [
            "--load-step-mw", "10",
            "--load-sine-amplitude", "0",
            "--load-sine-mean", "1",
            "--load-sine-freq-hz", "0",
            "--event-time", "10",
            "--event-duration", "60",
            "--load-ramp-on-s", "0.02",
            "--t-end", "70",
            "--dt", "0.002",
        ],
        "plot": [
            "--event-time", "10",
            "--event-duration", "60",
            "--t-min", "8",
            "--t-max", "45",
        ],
    },
    {
        # Slug-flow tower-SS forcing: zero-mean full-amplitude sine, +/-2 MW
        # at 0.234 Hz.
        "name": "Slugflow_0p234Hz_2MW",
        "sim": [
            "--load-step-mw", "2.0",
            "--load-sine-amplitude", "1.0",
            "--load-sine-mean", "0.0",
            "--load-sine-freq-hz", "0.234",
            "--event-time", "10",
            "--event-duration", "110",
            "--t-end", "120",
            "--dt", "0.002",
        ],
        "plot": [
            "--event-time", "10",
            "--event-duration", "110",
            "--t-min", "20",
            "--t-max", "60",
        ],
    },
    {
        # Process torsion forcing: zero-mean full-amplitude sine, +/-0.5 MW
        # at 3.49 Hz.
        "name": "Process_3p49Hz_0p5MW",
        "sim": [
            "--load-step-mw", "0.5",
            "--load-sine-amplitude", "1.0",
            "--load-sine-mean", "0.0",
            "--load-sine-freq-hz", "3.49",
            "--event-time", "10",
            "--event-duration", "30",
            "--t-end", "40",
            "--dt", "0.002",
        ],
        "plot": [
            "--event-time", "10",
            "--event-duration", "30",
            "--t-min", "14",
            "--t-max", "17",
        ],
    },
]


def run(cmd: list[str]) -> None:
    """Run a subprocess and raise if it fails."""
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    results: list[tuple[str, str, str]] = []
    all_start = time.perf_counter()

    for sc in SCENARIOS:
        name = sc["name"]
        outdir = DIAG_PLOTS / name
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== {name}: diagnostics simulation =====", flush=True)
        try:
            run([PY, "-u", DIAG, *sc["sim"], *DROOP])

            print(f"===== {name}: plotting =====", flush=True)
            run([PY, "-u", PLOT, "--output-dir", str(outdir), *sc["plot"]])

            for base in CSV_NAMES:
                shutil.copy2(CSV_DIR / base, outdir / base)

            results.append((name, "OK", str(outdir)))
            print(f"===== {name}: done -> {outdir} =====", flush=True)
        except subprocess.CalledProcessError as exc:
            results.append((name, f"FAILED ({exc.returncode})", str(outdir)))
            print(f"===== {name}: FAILED: {exc} =====", flush=True)

    print("\n================ summary ================", flush=True)
    for name, status, outdir in results:
        print(f"  {name:<22} {status:<14} {outdir}", flush=True)
    print(f"Total runtime: {time.perf_counter() - all_start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
