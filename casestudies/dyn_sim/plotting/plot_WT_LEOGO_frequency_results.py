"""
Create a full standalone-WT-style plot suite for one LEOGO frequency simulation.

Expected simulation outputs:
    casestudies/dyn_sim/results/WT1_LEOGO_frequency_results.csv
    casestudies/dyn_sim/results/WT1_LEOGO_frequency_droop_results.csv

This script supports both:
  - no-droop results
  - droop results

It creates frequency, WT drivetrain, MPT torque, WT power, UIC voltage/current,
UIC P/Q actual-versus-reference, synchronous-generator, and droop plots.

Place this file in:
    casestudies/dyn_sim/plotting/plot_WT_LEOGO_frequency_results.py

Run from the project root:

    # No droop
    python casestudies/dyn_sim/plotting/plot_WT_LEOGO_frequency_results.py

    # With droop
    python casestudies/dyn_sim/plotting/plot_WT_LEOGO_frequency_results.py `
        --input casestudies/dyn_sim/results/WT1_LEOGO_frequency_droop_results.csv `
        --output-dir casestudies/dyn_sim/logs/wt/plots/thesis/LEOGO_frequency_droop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Script location:
# <project-root>/casestudies/dyn_sim/plotting/plot_WT_LEOGO_frequency_results.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "casestudies"
    / "dyn_sim"
    / "results"
    / "WT1_LEOGO_frequency_results.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "casestudies"
    / "dyn_sim"
    / "logs"
    / "wt"
    / "plots"
    / "thesis"
    / "LEOGO_frequency"
)


def available_signals(
    df: pd.DataFrame,
    signals: list[tuple[str, str]],
    plot_name: str,
) -> list[tuple[str, str]]:
    """
    Return signals that exist in the CSV and report any missing optional signals.

    A plot is skipped only when none of its requested signals are available.
    """
    present = [(column, label) for column, label in signals if column in df.columns]
    missing = [column for column, _ in signals if column not in df.columns]

    if missing:
        print(f"{plot_name}: optional column(s) not found: {', '.join(missing)}")

    if not present:
        print(f"Skipping {plot_name}: none of the requested columns are available.")

    return present


def add_event_markers(
    ax: plt.Axes,
    event_time: float | None,
    event_duration: float | None,
) -> None:
    """Show load application and clearing time, when enabled."""
    if event_time is None:
        return

    ax.axvline(
        event_time,
        linestyle=":",
        linewidth=1.0,
        label="Load step starts",
    )

    if event_duration is not None:
        ax.axvline(
            event_time + event_duration,
            linestyle=":",
            linewidth=1.0,
            label="Load step clears",
        )


def save_plot(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str,
    ylabel: str,
    signals: list[tuple[str, str]],
    *,
    event_time: float | None,
    event_duration: float | None,
    title: str | None = None,
    nominal_line: float | None = None,
    nominal_label: str | None = None,
    zero_line: bool = False,
) -> None:
    """Save a single plot from one CSV result file."""
    signals_to_plot = available_signals(df, signals, filename)
    if not signals_to_plot:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for column, label in signals_to_plot:
        ax.plot(df["t"], df[column], label=label)

    if nominal_line is not None:
        ax.axhline(
            nominal_line,
            linestyle="--",
            linewidth=1.0,
            label=nominal_label or "Reference",
        )

    if zero_line:
        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
            label="Zero",
        )

    add_event_markers(ax, event_time, event_duration)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(True)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    fig.tight_layout()
    output_file = output_dir / filename
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


def add_rocof(df: pd.DataFrame) -> pd.DataFrame:
    """Add numerical RoCoF in Hz/s when time and frequency data are valid."""
    df = df.copy()

    if "f_grid_hz" not in df.columns:
        return df

    time_values = df["t"].to_numpy(dtype=float)
    frequency_values = df["f_grid_hz"].to_numpy(dtype=float)

    if len(time_values) < 3 or np.any(np.diff(time_values) <= 0.0):
        print("RoCoF not calculated: time samples must be strictly increasing.")
        return df

    df["rocof_hz_per_s"] = np.gradient(frequency_values, time_values)
    return df


def add_tracking_error(df: pd.DataFrame) -> pd.DataFrame:
    """Add UIC active-power tracking error on system base."""
    df = df.copy()

    p_ref_column = "P_ref_sys_pu"

    if "P_uic_bus_actual_sys_pu" in df.columns:
        p_uic_column = "P_uic_bus_actual_sys_pu"
    elif "P_uic_bus_sys_pu" in df.columns:
        p_uic_column = "P_uic_bus_sys_pu"
    else:
        return df

    if p_ref_column in df.columns:
        df["P_tracking_error_sys_pu"] = (
            df[p_ref_column] - df[p_uic_column]
        )

    return df


def add_unwrapped_current_angle(df: pd.DataFrame) -> pd.DataFrame:
    """Add a continuous UIC current angle for visualisation only."""
    df = df.copy()

    if "i_a_angle_deg" not in df.columns:
        return df

    angle_deg = df["i_a_angle_deg"].to_numpy(dtype=float)
    if not np.all(np.isfinite(angle_deg)):
        print("Current angle not unwrapped: non-finite samples found.")
        return df

    df["i_a_angle_unwrapped_deg"] = np.rad2deg(
        np.unwrap(np.deg2rad(angle_deg))
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot full WT/UIC/LEOGO frequency-response results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV from a no-droop or droop LEOGO frequency simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for generated PNG files.",
    )
    parser.add_argument(
        "--nominal-frequency",
        type=float,
        default=50.0,
        help="Nominal grid frequency in Hz.",
    )
    parser.add_argument(
        "--event-time",
        type=float,
        default=5.0,
        help="Load-step start time in seconds. Use a negative value to hide markers.",
    )
    parser.add_argument(
        "--event-duration",
        type=float,
        default=15.0,
        help="Load-step duration in seconds. Default event: 5 s to 20 s.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the grid-frequency plot after files have been saved.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Could not find input CSV:\n{args.input}\n\n"
            "Run the associated simulation first, or pass --input <path>."
        )

    df = pd.read_csv(args.input)

    if "t" not in df.columns:
        raise KeyError(
            "The input CSV must contain a 't' column. "
            f"Columns found: {list(df.columns)}"
        )

    df = add_rocof(df)
    df = add_tracking_error(df)
    df = add_unwrapped_current_angle(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_time = args.event_time if args.event_time >= 0.0 else None
    event_duration = args.event_duration if event_time is not None else None

    # -----------------------------------------------------------------
    # 1. Grid-frequency response
    # -----------------------------------------------------------------
    save_plot(
        df,
        args.output_dir,
        "frequency_grid_hz.png",
        "Grid frequency [Hz]",
        [("f_grid_hz", "Grid frequency")],
        event_time=event_time,
        event_duration=event_duration,
        title="Grid frequency",
        nominal_line=args.nominal_frequency,
        nominal_label=f"Nominal frequency ({args.nominal_frequency:g} Hz)",
    )

    if "f_grid_hz" in df.columns:
        df["frequency_deviation_hz"] = (
            df["f_grid_hz"] - args.nominal_frequency
        )

    save_plot(
        df,
        args.output_dir,
        "frequency_deviation_hz.png",
        "Frequency deviation [Hz]",
        [("frequency_deviation_hz", "Δf")],
        event_time=event_time,
        event_duration=event_duration,
        title="Frequency deviation",
        zero_line=True,
    )

    save_plot(
        df,
        args.output_dir,
        "rocof_hz_per_s.png",
        "RoCoF [Hz/s]",
        [("rocof_hz_per_s", "RoCoF")],
        event_time=event_time,
        event_duration=event_duration,
        title="Rate of change of frequency",
        zero_line=True,
    )

    # -----------------------------------------------------------------
    # 2. WT power and drivetrain response
    # -----------------------------------------------------------------
    save_plot(
        df,
        args.output_dir,
        "wt_power_pu.png",
        "Power [pu on system base]",
        [
            ("P_aero_sys_pu", "Aerodynamic power"),
            ("P_e_sys_pu", "WT electrical power"),
            ("P_ref_sys_pu", "WT power reference"),
            ("P_ref_instant_sys_pu", "WT instantaneous reference"),
            ("P_mpt_available_sys_pu", "Available MPT power"),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="WT active-power response",
    )

    save_plot(
        df,
        args.output_dir,
        "wt_frequency_support_power_pu.png",
        "Power [pu on system base]",
        [
            ("P_ref_sys_pu", "WT power reference"),
            ("P_uic_bus_actual_sys_pu", "UIC bus active power"),
            ("P_e_sys_pu", "WT electrical power"),
            ("P_aero_sys_pu", "Aerodynamic power"),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="WT frequency-support response",
    )

    save_plot(
        df,
        args.output_dir,
        "wt_power_tracking_error_pu.png",
        "Power error [pu on system base]",
        [("P_tracking_error_sys_pu", "P_ref - P_UIC")],
        event_time=event_time,
        event_duration=event_duration,
        title="UIC active-power tracking error",
        zero_line=True,
    )

    save_plot(
        df,
        args.output_dir,
        "wt_rotor_speeds_pu.png",
        "Speed [pu]",
        [
            ("omega_m_pu", "Mechanical speed ω_m"),
            ("omega_e_pu", "Electrical speed ω_e"),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="WT drivetrain speeds",
        nominal_line=1.0,
        nominal_label="1 pu",
    )

    save_plot(
        df,
        args.output_dir,
        "wt_T_mpt_pu.png",
        "Mechanical torque [pu on WT base]",
        [("T_mpt_wt_pu", "MPT mechanical torque")],
        event_time=event_time,
        event_duration=event_duration,
        title="MPT mechanical torque",
    )

    save_plot(
        df,
        args.output_dir,
        "wt_pitch_deg.png",
        "Pitch angle [deg]",
        [("pitch_deg", "Pitch angle")],
        event_time=event_time,
        event_duration=event_duration,
        title="WT pitch angle",
    )

    save_plot(
        df,
        args.output_dir,
        "wind_speed_mps.png",
        "Wind speed [m/s]",
        [("wind_speed_mps", "Wind speed")],
        event_time=event_time,
        event_duration=event_duration,
        title="Wind speed",
    )

    # -----------------------------------------------------------------
    # 3. UIC terminal/internal electrical response
    # -----------------------------------------------------------------
    save_plot(
        df,
        args.output_dir,
        "wt_voltages_pu.png",
        "Voltage [pu]",
        [
            ("V_WTG1_LV_pu", "UIC terminal voltage"),
            ("vi_mag_pu", "UIC internal voltage"),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="UIC terminal and internal voltage",
        nominal_line=1.0,
        nominal_label="1 pu",
    )

    save_plot(
        df,
        args.output_dir,
        "uic_bus_PQ_pu.png",
        "Power [pu on system base]",
        [
            ("P_uic_bus_actual_sys_pu", "UIC P, actual"),
            ("Q_uic_bus_actual_sys_pu", "UIC Q, actual"),
            ("P_uic_bus_ref_sys_pu", "UIC P, reference"),
            ("Q_uic_bus_ref_sys_pu", "UIC Q, reference"),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="UIC bus-side active and reactive power",
    )

    save_plot(
        df,
        args.output_dir,
        "uic_current_pu.png",
        "Current magnitude [pu on UIC base]",
        [
            ("i_a_mag_pu_uic", "UIC current"),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="UIC current magnitude",
    )

    save_plot(
        df,
        args.output_dir,
        "uic_current_angle_deg.png",
        "Current angle [deg]",
        [("i_a_angle_unwrapped_deg", "UIC current angle")],
        event_time=event_time,
        event_duration=event_duration,
        title="UIC current angle",
    )

    # -----------------------------------------------------------------
    # 4. LEOGO synchronous-generator response
    # -----------------------------------------------------------------
    save_plot(
        df,
        args.output_dir,
        "sync_generators_PQ_pu.png",
        "Power [pu on system base]",
        [
            (
                "P_sync_generators_total_sys_pu",
                "Synchronous-generator active power",
            ),
            (
                "Q_sync_generators_total_sys_pu",
                "Synchronous-generator reactive power",
            ),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="Aggregate synchronous-generator response",
    )

    # -----------------------------------------------------------------
    # 5. Droop-specific figures
    #
    # In the no-droop result file these signals are normally present but
    # P_droop_delta_uic_pu is zero. The plots therefore remain useful.
    # -----------------------------------------------------------------
    save_plot(
        df,
        args.output_dir,
        "wt_droop_command_uic_pu.png",
        "Active power [pu on UIC base]",
        [
            ("P_base_uic_pu", "Base WT reference"),
            ("P_ref_uic_pu", "Total WT reference"),
            ("P_available_uic_pu", "Available WT power"),
        ],
        event_time=event_time,
        event_duration=event_duration,
        title="WT power command and available power",
    )

    save_plot(
        df,
        args.output_dir,
        "wt_droop_delta_uic_pu.png",
        "Droop contribution [pu on UIC base]",
        [("P_droop_delta_uic_pu", "ΔP droop")],
        event_time=event_time,
        event_duration=event_duration,
        title="WT droop contribution",
        zero_line=True,
    )

    print(f"\nPlots saved to:\n{args.output_dir}")

    if args.show and "f_grid_hz" in df.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(df["t"], df["f_grid_hz"], label="Grid frequency")
        ax.axhline(
            args.nominal_frequency,
            linestyle="--",
            linewidth=1.0,
            label="Nominal frequency",
        )
        add_event_markers(ax, event_time, event_duration)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Grid frequency [Hz]")
        ax.grid(True)
        ax.legend()
        
        plt.show()


if __name__ == "__main__":
    main()
