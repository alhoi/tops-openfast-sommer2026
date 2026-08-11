"""
Plot electromechanical diagnostic results for matched LEOGO wind-turbine
simulations with and without frequency droop control.

The script reads diagnostic/measurement CSV files only. It never changes or
overwrites the CSV files. Generated figures are written to a separate folder.

By default, the script reads exactly these two diagnostic CSV files:
  casestudies/dyn_sim/results/WT1_LEOGO_frequency_1MW_no_droop_deLoaded_MEASUREMENTS.csv
  casestudies/dyn_sim/results/WT1_LEOGO_frequency_1MW_droop_deLoaded_MEASUREMENTS.csv

Use --no-droop-csv and --droop-csv only to override these paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# This script is in casestudies/dyn_sim/plotting, so parents[3] is
# the repository root. parents[2] would incorrectly resolve to
# <repository>/casestudies and duplicate "casestudies" in file paths.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results"


REQUIRED_COLUMNS = {
    "t",
    "omega_m_pu",
    "omega_e_pu",
    "omega_e_filt_pu",
    "omega_e_control_pu",
    "omega_ref_pu",
    "omega_speed_error_pu",
    "T_a_wt_pu",
    "T_e_wt_pu",
    "T_shaft_wt_pu",
    "pitch_deg",
    "pitch_reference_deg",
    "pitch_rate_deg_s",
    "P_aero_sys_pu",
    "P_base_uic_pu",
    "P_available_uic_pu",
    "P_ref_uic_pu",
    "P_uic_bus_actual_sys_pu",
    "P_uic_bus_ref_sys_pu",
    "P_e_sys_pu",
    "V_WTG1_LV_pu",
    "vi_mag_pu",
    "I_uic_pu",
    "i_a_angle_deg",
    "P_droop_delta_uic_pu",
}


def load_diagnostic_csv(path: Path, case_label: str) -> pd.DataFrame:
    """Read and validate one diagnostic CSV."""
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise KeyError(
            f"{path.name} is missing diagnostic columns:\n"
            f"  {', '.join(missing)}\n\n"
            "Use the diagnostic simulation runner that writes torque, "
            "filtered-speed and pitch-controller columns."
        )

    df = df.sort_values("t").reset_index(drop=True)
    return df


def add_event_markers(ax: plt.Axes, event_time: float, event_duration: float) -> None:
    """Show load application and load removal without modifying data."""
    ax.axvline(event_time, linestyle=":", linewidth=1.5)
    ax.axvline(event_time + event_duration, linestyle=":", linewidth=1.5)


def finish_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    output_path: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Apply shared formatting and save a figure."""
    add_event_markers(ax, event_time, event_duration)
    if t_min is not None or t_max is not None:
        ax.set_xlim(left=t_min, right=t_max)

    ax.set_xlabel("Time [s]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def uic_to_system_base_scale(df: pd.DataFrame) -> float:
    """
    Infer the UIC-to-system-base conversion from logged reference columns.

    P_ref_sys_pu / P_ref_uic_pu is normally 20/100 = 0.2 in this model.
    Inferring it avoids hard-coding the system base in the plotting script.
    """
    if "P_ref_sys_pu" in df.columns:
        mask = np.abs(df["P_ref_uic_pu"].to_numpy(dtype=float)) > 1e-10
        if np.any(mask):
            ratios = (
                df.loc[mask, "P_ref_sys_pu"].to_numpy(dtype=float)
                / df.loc[mask, "P_ref_uic_pu"].to_numpy(dtype=float)
            )
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                return float(np.median(ratios))

    # Current LEOGO setup: 20 MVA UIC base / 100 MVA system base.
    return 0.2


def add_power_chain_system_base(df: pd.DataFrame) -> pd.DataFrame:
    """Create system-base versions of UIC-base command signals for plotting."""
    result = df.copy()
    scale = uic_to_system_base_scale(result)
    result["P_base_sys_pu_plot"] = result["P_base_uic_pu"] * scale
    result["P_available_sys_pu_plot"] = result["P_available_uic_pu"] * scale
    result["P_ref_sys_pu_plot"] = result["P_ref_uic_pu"] * scale
    result["P_droop_sys_pu_plot"] = result["P_droop_delta_uic_pu"] * scale
    result["T_rotor_net_wt_pu_plot"] = (
        result["T_a_wt_pu"] - result["T_shaft_wt_pu"]
    )
    result["T_generator_net_wt_pu_plot"] = (
        result["T_shaft_wt_pu"] - result["T_e_wt_pu"]
    )
    return result


def plot_uic_power_comparison(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Compare actual UIC active power and its reference."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        no_droop["t"],
        no_droop["P_uic_bus_actual_sys_pu"],
        label="UIC active power, without droop",
    )
    ax.plot(
        droop["t"],
        droop["P_uic_bus_actual_sys_pu"],
        label="UIC active power, with droop",
    )
    ax.plot(
        no_droop["t"],
        no_droop["P_ref_sys_pu_plot"],
        linestyle="--",
        label="WT reference, without droop",
    )
    ax.plot(
        droop["t"],
        droop["P_ref_sys_pu_plot"],
        linestyle="--",
        label="WT reference, with droop",
    )
    ax.set_title("UIC active-power comparison")
    ax.set_ylabel("Active power [pu on system base]")
    finish_figure(
        fig, ax, output_dir / "01_uic_active_power_comparison.png",
        event_time, event_duration, t_min, t_max,
    )


def plot_droop_contribution_comparison(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Compare the droop contribution on a common system base."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        no_droop["t"],
        no_droop["P_droop_sys_pu_plot"],
        label="Without droop",
    )
    ax.plot(
        droop["t"],
        droop["P_droop_sys_pu_plot"],
        label="With droop",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Zero")
    ax.set_title("Additional active power from droop control")
    ax.set_ylabel("Droop contribution [pu on system base]")
    finish_figure(
        fig, ax, output_dir / "02_droop_contribution_comparison.png",
        event_time, event_duration, t_min, t_max,
    )



def plot_uic_voltage_comparison(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """
    Compare the terminal-bus voltage with the UIC internal-voltage state.

    V_WTG1_LV_pu is |V_t| at Busbar WTG1 LV.
    vi_mag_pu is |v_i| where v_i = vi_x + j vi_y is the UIC internal state.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        no_droop["t"],
        no_droop["V_WTG1_LV_pu"],
        label=r"$|V_t|$, without droop",
    )
    ax.plot(
        no_droop["t"],
        no_droop["vi_mag_pu"],
        linestyle="--",
        label=r"$|v_i|$, without droop",
    )
    ax.plot(
        droop["t"],
        droop["V_WTG1_LV_pu"],
        label=r"$|V_t|$, with droop",
    )
    ax.plot(
        droop["t"],
        droop["vi_mag_pu"],
        linestyle="--",
        label=r"$|v_i|$, with droop",
    )
    ax.set_title("UIC terminal and internal-voltage magnitudes")
    ax.set_ylabel("Voltage magnitude [pu]")
    finish_figure(
        fig, ax, output_dir / "10_uic_voltage_comparison.png",
        event_time, event_duration, t_min, t_max,
    )


def plot_uic_current_comparison(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Compare magnitude of current through the UIC reactance xf."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        no_droop["t"],
        no_droop["I_uic_pu"],
        label=r"$|I_a|$, without droop",
    )
    ax.plot(
        droop["t"],
        droop["I_uic_pu"],
        label=r"$|I_a|$, with droop",
    )
    ax.set_title("UIC current magnitude")
    ax.set_ylabel("Current magnitude [pu on UIC base]")
    finish_figure(
        fig, ax, output_dir / "11_uic_current_magnitude_comparison.png",
        event_time, event_duration, t_min, t_max,
    )


def plot_uic_current_angle_comparison(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """
    Compare current phase angle.

    This angle is expressed in the network reference frame. It is useful for
    transient diagnosis, but may wrap by +/-180 degrees in other operating
    conditions.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        no_droop["t"],
        no_droop["i_a_angle_deg"],
        label=r"$\angle I_a$, without droop",
    )
    ax.plot(
        droop["t"],
        droop["i_a_angle_deg"],
        label=r"$\angle I_a$, with droop",
    )
    ax.set_title("UIC current angle")
    ax.set_ylabel("Current angle [deg]")
    finish_figure(
        fig, ax, output_dir / "12_uic_current_angle_comparison.png",
        event_time, event_duration, t_min, t_max,
    )


def plot_power_location_comparison(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """
    Compare bus-side UIC power with the WT electrical-power signal used in Te.

    P_e_sys_pu is the WT-model input used in the electromagnetic-torque
    calculation. P_uic_bus_actual_sys_pu is the actual complex-power result
    at Busbar WTG1 LV. They are deliberately plotted separately.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        no_droop["t"],
        no_droop["P_uic_bus_actual_sys_pu"],
        label=r"$P_{\mathrm{UIC,bus}}$, without droop",
    )
    ax.plot(
        no_droop["t"],
        no_droop["P_e_sys_pu"],
        linestyle="--",
        label=r"$P_{e,\mathrm{WT}}$, without droop",
    )
    ax.plot(
        droop["t"],
        droop["P_uic_bus_actual_sys_pu"],
        label=r"$P_{\mathrm{UIC,bus}}$, with droop",
    )
    ax.plot(
        droop["t"],
        droop["P_e_sys_pu"],
        linestyle="--",
        label=r"$P_{e,\mathrm{WT}}$, with droop",
    )
    ax.set_title("Bus-side UIC power and WT torque-power input")
    ax.set_ylabel("Active power [pu on system base]")
    finish_figure(
        fig, ax, output_dir / "13_bus_power_vs_wt_torque_power.png",
        event_time, event_duration, t_min, t_max,
    )


def plot_bus_power_tracking_comparison(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """
    Compare UIC bus-side active power with the bus-side reference.

    This isolates whether the initial overshoot is due to the UIC/network
    response rather than the slowly changing WT reference.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        no_droop["t"],
        no_droop["P_uic_bus_actual_sys_pu"],
        label=r"$P_{\mathrm{UIC,bus}}$, without droop",
    )
    ax.plot(
        no_droop["t"],
        no_droop["P_uic_bus_ref_sys_pu"],
        linestyle="--",
        label=r"$P_{\mathrm{UIC,bus,ref}}$, without droop",
    )
    ax.plot(
        droop["t"],
        droop["P_uic_bus_actual_sys_pu"],
        label=r"$P_{\mathrm{UIC,bus}}$, with droop",
    )
    ax.plot(
        droop["t"],
        droop["P_uic_bus_ref_sys_pu"],
        linestyle="--",
        label=r"$P_{\mathrm{UIC,bus,ref}}$, with droop",
    )
    ax.set_title("UIC bus-side active power and bus-side reference")
    ax.set_ylabel("Active power [pu on system base]")
    finish_figure(
        fig, ax, output_dir / "14_uic_bus_power_tracking_comparison.png",
        event_time, event_duration, t_min, t_max,
    )


def plot_bus_vs_wt_power_difference(
    no_droop: pd.DataFrame,
    droop: pd.DataFrame,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """
    Plot P_UIC,bus - P_e,WT directly.

    A non-zero value does not automatically imply physical loss. The two
    signals are produced at different model interfaces and are relevant to
    different equations. The plot identifies when their transient behaviour
    differs and therefore needs closer model-level inspection.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    delta_no_droop = (
        no_droop["P_uic_bus_actual_sys_pu"]
        - no_droop["P_e_sys_pu"]
    )
    delta_droop = (
        droop["P_uic_bus_actual_sys_pu"]
        - droop["P_e_sys_pu"]
    )
    ax.plot(
        no_droop["t"],
        delta_no_droop,
        label=r"$P_{\mathrm{UIC,bus}}-P_{e,\mathrm{WT}}$, without droop",
    )
    ax.plot(
        droop["t"],
        delta_droop,
        label=r"$P_{\mathrm{UIC,bus}}-P_{e,\mathrm{WT}}$, with droop",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Zero")
    ax.set_title("Difference between bus power and WT torque-power input")
    ax.set_ylabel("Power difference [pu on system base]")
    finish_figure(
        fig, ax, output_dir / "15_bus_minus_wt_torque_power.png",
        event_time, event_duration, t_min, t_max,
    )

def plot_power_chain(
    df: pd.DataFrame,
    case_title: str,
    filename: str,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """
    Plot aerodynamic, electrical, available and commanded powers.

    All curves are shown on the system base to allow direct comparison.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["t"], df["P_aero_sys_pu"], label=r"$P_{\mathrm{aero}}$")
    ax.plot(
        df["t"],
        df["P_uic_bus_actual_sys_pu"],
        label=r"$P_{\mathrm{UIC}}$ (actual)",
    )
    ax.plot(
        df["t"],
        df["P_base_sys_pu_plot"],
        linestyle="--",
        label=r"$P_{\mathrm{base}}$",
    )
    ax.plot(
        df["t"],
        df["P_available_sys_pu_plot"],
        linestyle="--",
        label=r"$P_{\mathrm{available}}$",
    )
    ax.plot(
        df["t"],
        df["P_ref_sys_pu_plot"],
        linestyle="--",
        label=r"$P_{\mathrm{ref}}$",
    )
    ax.set_title(f"Power chain: {case_title}")
    ax.set_ylabel("Power [pu on system base]")
    finish_figure(
        fig, ax, output_dir / filename,
        event_time, event_duration, t_min, t_max,
    )


def plot_torque_balance(
    df: pd.DataFrame,
    case_title: str,
    filename: str,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot aerodynamic, shaft and electromagnetic torque on the WT base."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["t"], df["T_a_wt_pu"], label=r"$T_a$")
    ax.plot(df["t"], df["T_shaft_wt_pu"], label=r"$T_s$")
    ax.plot(df["t"], df["T_e_wt_pu"], label=r"$T_e$")
    ax.set_title(f"Drivetrain torque balance: {case_title}")
    ax.set_ylabel("Torque [pu on WT base]")
    finish_figure(
        fig, ax, output_dir / filename,
        event_time, event_duration, t_min, t_max,
    )


def plot_net_torque(
    df: pd.DataFrame,
    case_title: str,
    filename: str,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot net torque on the rotor and generator inertias."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        df["t"],
        df["T_rotor_net_wt_pu_plot"],
        label=r"$T_a-T_s$ (rotor net torque)",
    )
    ax.plot(
        df["t"],
        df["T_generator_net_wt_pu_plot"],
        label=r"$T_s-T_e$ (generator net torque)",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Zero")
    ax.set_title(f"Net torque driving the drivetrain inertias: {case_title}")
    ax.set_ylabel("Torque [pu on WT base]")
    finish_figure(
        fig, ax, output_dir / filename,
        event_time, event_duration, t_min, t_max,
    )


def plot_speeds(
    df: pd.DataFrame,
    case_title: str,
    filename: str,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot mechanical, raw electrical and filtered electrical speeds."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["t"], df["omega_m_pu"], label=r"$\omega_m$")
    ax.plot(df["t"], df["omega_e_pu"], label=r"$\omega_e$")
    ax.plot(
        df["t"],
        df["omega_e_filt_pu"],
        linestyle="--",
        label=r"$\omega_{e,\mathrm{filt}}$",
    )
    ax.plot(
        df["t"],
        df["omega_ref_pu"],
        linestyle="--",
        label=r"$\omega_{\mathrm{ref}}$",
    )
    ax.set_title(f"Drivetrain and controller speeds: {case_title}")
    ax.set_ylabel("Speed [pu]")
    finish_figure(
        fig, ax, output_dir / filename,
        event_time, event_duration, t_min, t_max,
    )


def plot_speed_error(
    df: pd.DataFrame,
    case_title: str,
    filename: str,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot the exact speed error supplied to the pitch controller."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        df["t"],
        df["omega_speed_error_pu"],
        label=r"$\omega_{e,\mathrm{filt}}-\omega_{\mathrm{ref}}$",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Zero")
    ax.set_title(f"Pitch-controller speed error: {case_title}")
    ax.set_ylabel("Speed error [pu]")
    finish_figure(
        fig, ax, output_dir / filename,
        event_time, event_duration, t_min, t_max,
    )


def plot_pitch_position(
    df: pd.DataFrame,
    case_title: str,
    filename: str,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot pitch angle and pitch actuator reference."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df["t"], df["pitch_deg"], label=r"$\beta$ (actual)")
    ax.plot(
        df["t"],
        df["pitch_reference_deg"],
        linestyle="--",
        label=r"$\beta_{\mathrm{ref}}$",
    )
    ax.set_title(f"Pitch position and reference: {case_title}")
    ax.set_ylabel("Pitch angle [deg]")
    finish_figure(
        fig, ax, output_dir / filename,
        event_time, event_duration, t_min, t_max,
    )


def plot_pitch_rate(
    df: pd.DataFrame,
    case_title: str,
    filename: str,
    output_dir: Path,
    event_time: float,
    event_duration: float,
    t_min: float | None,
    t_max: float | None,
) -> None:
    """Plot the pitch-servo derivative after rate limiting."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        df["t"],
        df["pitch_rate_deg_s"],
        label=r"$\dot{\beta}$",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, label="Zero")
    ax.set_title(f"Pitch-actuator rate: {case_title}")
    ax.set_ylabel("Pitch rate [deg/s]")
    finish_figure(
        fig, ax, output_dir / filename,
        event_time, event_duration, t_min, t_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create comparison plots from wind-turbine diagnostic CSV files. "
            "The CSV input files are never modified."
        )
    )
    parser.add_argument(
        "--no-droop-csv",
        type=Path,
        default=None,
        help="Path to diagnostic CSV for the no-droop case.",
    )
    parser.add_argument(
        "--droop-csv",
        type=Path,
        default=None,
        help="Path to diagnostic CSV for the droop case.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "diagnostic_plots" / "Sine_1.2Hz.2.5",
        help="Directory for PNG figures.",
    )
    parser.add_argument("--event-time", type=float, default=5.0)
    parser.add_argument("--event-duration", type=float, default=50.0)
    parser.add_argument(
        "--t-min",
        type=float,
        default=None,
        help="Optional left x-axis limit for event zoom plots.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Optional right x-axis limit for event zoom plots.",
    )
    args = parser.parse_args()

    # Read exactly the two measurement files produced by the diagnostic
    # simulation. Command-line paths are optional overrides only.
    no_droop_path = (
        args.no_droop_csv
        if args.no_droop_csv is not None
        else (
            RESULTS_DIR
            / "csv_files"
            / "WT1_LEOGO_frequency_Sine_freq_NOdroop.csv"
        )
    )
    droop_path = (
        args.droop_csv
        if args.droop_csv is not None
        else (
            RESULTS_DIR
            / "csv_files"
            / "WT1_LEOGO_frequency_Sine_freq_droop.csv"
        )
    )

    for case_label, csv_path in (
        ("no-droop", no_droop_path),
        ("droop", droop_path),
    ):
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Could not find the {case_label} measurement CSV:\n"
                f"  {csv_path}\n\n"
                "Run test_WT_LEOGO_droop_comparison_diagnostics.py first, "
                "or pass an explicit CSV file with the corresponding "
                "command-line option."
            )

    no_droop = add_power_chain_system_base(
        load_diagnostic_csv(no_droop_path, "no-droop")
    )
    droop = add_power_chain_system_base(
        load_diagnostic_csv(droop_path, "droop")
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_uic_power_comparison(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )
    plot_droop_contribution_comparison(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )

    plot_uic_voltage_comparison(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )
    plot_uic_current_comparison(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )
    plot_uic_current_angle_comparison(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )
    plot_power_location_comparison(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )
    plot_bus_power_tracking_comparison(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )
    plot_bus_vs_wt_power_difference(
        no_droop, droop, output_dir,
        args.event_time, args.event_duration, args.t_min, args.t_max,
    )

    for df, title, stem in (
        (no_droop, "Without droop", "no_droop"),
        (droop, "With droop", "droop"),
    ):
        plot_power_chain(
            df, title, f"03_power_chain_{stem}.png", output_dir,
            args.event_time, args.event_duration, args.t_min, args.t_max,
        )
        plot_torque_balance(
            df, title, f"04_torque_balance_{stem}.png", output_dir,
            args.event_time, args.event_duration, args.t_min, args.t_max,
        )
        plot_net_torque(
            df, title, f"05_net_torque_{stem}.png", output_dir,
            args.event_time, args.event_duration, args.t_min, args.t_max,
        )
        plot_speeds(
            df, title, f"06_drivetrain_speeds_{stem}.png", output_dir,
            args.event_time, args.event_duration, args.t_min, args.t_max,
        )
        plot_speed_error(
            df, title, f"07_pitch_speed_error_{stem}.png", output_dir,
            args.event_time, args.event_duration, args.t_min, args.t_max,
        )
        plot_pitch_position(
            df, title, f"08_pitch_position_{stem}.png", output_dir,
            args.event_time, args.event_duration, args.t_min, args.t_max,
        )
        plot_pitch_rate(
            df, title, f"09_pitch_rate_{stem}.png", output_dir,
            args.event_time, args.event_duration, args.t_min, args.t_max,
        )

    print("Read no-droop CSV:", no_droop_path)
    print("Read droop CSV:   ", droop_path)
    print("Saved diagnostic plots to:", output_dir)
    print("CSV input files were not modified.")


if __name__ == "__main__":
    main()
