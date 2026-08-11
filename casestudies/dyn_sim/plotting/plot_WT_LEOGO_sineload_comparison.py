"""
Plot sinusoidal-load LEOGO wind-turbine results, comparing the cases with
and without frequency droop control.

This script is written specifically for the sinusoidal-load study produced by
    casestudies/dyn_sim/test_WT_LEOGO_droop_comparison_diagnostics.py

Differences from the older comparison plotter:
  * It reads the CSV files from results/csv_files/ (where the diagnostics
    runner actually writes them).
  * It plots the full run. No event-start / event-clear vertical markers are
    drawn, because the load is now a continuous sinusoidal event.

The CSV input files are only read, never modified. Figures are written to a
separate output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# This file is in casestudies/dyn_sim/plotting, so parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"

DEFAULT_NO_DROOP_CSV = CSV_DIR / "WT1_LEOGO_frequency_SineLoad_noDroop.csv"
DEFAULT_DROOP_CSV = CSV_DIR / "WT1_LEOGO_frequency_SineLoad_droop.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Read one result CSV and sort it by time."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find result CSV:\n  {path}\n\n"
            "Run test_WT_LEOGO_droop_comparison_diagnostics.py first, or pass "
            "an explicit path with --no-droop-csv / --droop-csv."
        )
    df = pd.read_csv(path)
    return df.sort_values("t").reset_index(drop=True)


def finish_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    output_path: Path,
    ylabel: str,
    title: str,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Apply shared formatting and save a figure (no event markers)."""
    if t_min is not None or t_max is not None:
        ax.set_xlim(left=t_min, right=t_max)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_compare(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot one column from both cases on the same axes over the full run."""
    if column not in no_droop.columns or column not in droop.columns:
        print(f"  Skipping '{column}' (not present in both CSV files).")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(no_droop["t"], no_droop[column], label="Without droop")
    ax.plot(droop["t"], droop[column], label="With droop")
    finish_figure(fig, ax, output_dir / filename, ylabel, title, t_min, t_max)


def plot_applied_load(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot the sinusoidal applied load (identical in both cases)."""
    if "load_step_mw" not in no_droop.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(no_droop["t"], no_droop["load_step_mw"], label="Applied load")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    finish_figure(
        fig, ax, output_dir / "00_applied_sine_load.png",
        "Applied load [MW]", "Sinusoidal applied load", t_min, t_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sinusoidal-load LEOGO wind-turbine results with and "
            "without droop control. CSV inputs are never modified."
        )
    )
    parser.add_argument("--no-droop-csv", type=Path, default=DEFAULT_NO_DROOP_CSV)
    parser.add_argument("--droop-csv", type=Path, default=DEFAULT_DROOP_CSV)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CSV_DIR.parent / "sineload_plots",
        help="Directory for PNG figures.",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Optional left x-axis limit. Default plots the full run.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional right x-axis limit. Default plots the full run.",
    )
    args = parser.parse_args()

    no_droop = load_csv(args.no_droop_csv)
    droop = load_csv(args.droop_csv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_applied_load(no_droop, droop, output_dir, args.t_min, args.t_max)

    # (column, y-label, title, output filename)
    panels = [
        (
            "f_grid_hz",
            "Frequency [Hz]",
            "Grid frequency (COI)",
            "01_grid_frequency.png",
        ),
        (
            "frequency_deviation_hz",
            "Frequency deviation [Hz]",
            "Grid frequency deviation from nominal",
            "02_frequency_deviation.png",
        ),
        (
            "P_ref_sys_pu",
            "Active power [pu on system base]",
            "WT active-power reference",
            "03_wt_power_reference.png",
        ),
        (
            "P_uic_bus_actual_sys_pu",
            "Active power [pu on system base]",
            "UIC bus-side active power",
            "04_uic_bus_active_power.png",
        ),
        (
            "P_droop_delta_uic_pu",
            "Droop contribution [pu on UIC base]",
            "Additional active power from droop control",
            "05_droop_contribution.png",
        ),
        (
            "omega_m_pu",
            "Speed [pu]",
            "Rotor (mechanical) speed",
            "06_rotor_speed.png",
        ),
        (
            "omega_e_pu",
            "Speed [pu]",
            "Generator (electrical-side) speed",
            "07_generator_speed.png",
        ),
        (
            "pitch_deg",
            "Pitch angle [deg]",
            "Blade pitch angle",
            "08_pitch_angle.png",
        ),
        (
            "V_WTG1_LV_pu",
            "Voltage magnitude [pu]",
            "Terminal voltage at Busbar WTG1 LV",
            "09_terminal_voltage.png",
        ),
        (
            "I_uic_pu",
            "Current magnitude [pu on UIC base]",
            "UIC current magnitude",
            "10_uic_current.png",
        ),
        (
            "P_sync_generators_total_sys_pu",
            "Active power [pu on system base]",
            "Total synchronous-generator active power",
            "11_sync_gen_power.png",
        ),
    ]

    for column, ylabel, title, filename in panels:
        plot_compare(
            no_droop, droop, column, ylabel, title, filename,
            output_dir, args.t_min, args.t_max,
        )

    print("Read no-droop CSV:", args.no_droop_csv)
    print("Read droop CSV:   ", args.droop_csv)
    print("Saved plots to:   ", output_dir)
    print("CSV input files were not modified.")


if __name__ == "__main__":
    main()
