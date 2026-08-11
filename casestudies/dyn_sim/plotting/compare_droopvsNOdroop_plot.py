"""
Overlay plots for LEOGO frequency simulations:
  - WT1_LEOGO_frequency_results.csv          (without droop)
  - WT1_LEOGO_frequency_droop_results.csv    (with droop)

Run from the project root:
    python casestudies/dyn_sim/plotting/compare_droopvsNOdroop_plot.py

The script creates overlay figures in:
    casestudies/dyn_sim/logs/wt/plots/thesis/LEOGO_frequency_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_column(df: pd.DataFrame, column: str, source_name: str) -> None:
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' is missing from {source_name}. "
            f"Available columns: {', '.join(df.columns)}"
        )


def add_event_markers(ax, event_time: float, event_duration: float, include_labels: bool = True) -> None:
    clear_time = event_time + event_duration
    if include_labels:
        ax.axvline(event_time, linestyle=":", label="Load step starts")
        ax.axvline(clear_time, linestyle=":", label="Load step clears")
    else:
        ax.axvline(event_time, linestyle=":")
        ax.axvline(clear_time, linestyle=":")


def save_figure(fig, output_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _finite_values(*series) -> np.ndarray:
    """Return one finite NumPy array from one or more pandas/NumPy series."""
    values = np.concatenate(
        [np.asarray(series_i, dtype=float).ravel() for series_i in series]
    )
    return values[np.isfinite(values)]


def _frequency_decimal_places(values: np.ndarray) -> int:
    """
    Pick readable absolute-Hz precision without Matplotlib's misleading
    additive offset, e.g. '1e-9 + 4.999999999e1'.
    """
    if values.size == 0:
        return 3

    span = float(np.ptp(values))

    if span < 1e-7:
        return 10
    if span < 1e-5:
        return 8
    if span < 1e-3:
        return 6
    if span < 1e-1:
        return 4
    return 3


def format_absolute_frequency_axis(ax) -> None:
    """
    Use a fixed, report-friendly frequency axis.

    This deliberately avoids auto-zooming around numerical noise. A zero-load
    test therefore appears as the expected straight line at 50 Hz.
    """
    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
        scilimits=(-10, 10),
    )
    ax.set_ylim(49.75, 50.25)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def plot_grid_frequency_comparison(
    df_no_droop,
    df_droop,
    output_dir,
    event_time,
    event_duration,
) -> None:
    """Frequency in actual Hz with fixed report-friendly y-limits."""
    ensure_column(df_no_droop, "f_grid_hz", "no-droop CSV")
    ensure_column(df_droop, "f_grid_hz", "droop CSV")

    f_no_droop = df_no_droop["f_grid_hz"].to_numpy(dtype=float)
    f_droop = df_droop["f_grid_hz"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], f_no_droop, label="Without droop")
    ax.plot(df_droop["t"], f_droop, label="With droop")
    ax.axhline(50.0, linestyle="--", label="Nominal value (50)")
    add_event_markers(ax, event_time, event_duration)

    format_absolute_frequency_axis(ax)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Grid frequency [Hz]")
    ax.set_title("Grid-frequency comparison")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_frequency_grid_hz.png")


def plot_frequency_deviation_comparison(
    df_no_droop,
    df_droop,
    output_dir,
    event_time,
    event_duration,
) -> None:
    """Frequency deviation in actual Hz with fixed report-friendly limits."""
    ensure_column(df_no_droop, "frequency_deviation_hz", "no-droop CSV")
    ensure_column(df_droop, "frequency_deviation_hz", "droop CSV")

    dev_no_droop = df_no_droop["frequency_deviation_hz"].to_numpy(dtype=float)
    dev_droop = df_droop["frequency_deviation_hz"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df_no_droop["t"],
        dev_no_droop,
        label="Without droop",
    )
    ax.plot(
        df_droop["t"],
        dev_droop,
        label="With droop",
    )
    ax.axhline(0.0, linestyle="--", label="Zero")
    add_event_markers(ax, event_time, event_duration)

    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
        scilimits=(-10, 10),
    )
    ax.set_ylim(-0.10, 0.10)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency deviation [Hz]")
    ax.set_title("Frequency-deviation comparison")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_frequency_deviation_hz.png")

def plot_two_cases(
    df_no_droop: pd.DataFrame,
    df_droop: pd.DataFrame,
    column: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
    event_time: float,
    event_duration: float,
    title: str | None = None,
    zero_line: bool = False,
    nominal_line: float | None = None,
) -> None:
    ensure_column(df_no_droop, column, "no-droop CSV")
    ensure_column(df_droop, column, "droop CSV")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], df_no_droop[column], label="Without droop")
    ax.plot(df_droop["t"], df_droop[column], label="With droop")

    if nominal_line is not None:
        ax.axhline(nominal_line, linestyle="--", label=f"Nominal value ({nominal_line:g})")
    if zero_line:
        ax.axhline(0.0, linestyle="--", label="Zero")

    add_event_markers(ax, event_time, event_duration)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, filename)


def plot_power_comparison(df_no_droop, df_droop, output_dir, event_time, event_duration) -> None:
    for col in ("P_ref_sys_pu", "P_uic_bus_sys_pu"):
        ensure_column(df_no_droop, col, "no-droop CSV")
        ensure_column(df_droop, col, "droop CSV")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], df_no_droop["P_ref_sys_pu"], label="WT reference, without droop")
    ax.plot(df_no_droop["t"], df_no_droop["P_uic_bus_sys_pu"], label="UIC P, without droop")
    ax.plot(df_droop["t"], df_droop["P_ref_sys_pu"], label="WT reference, with droop")
    ax.plot(df_droop["t"], df_droop["P_uic_bus_sys_pu"], label="UIC P, with droop")
    add_event_markers(ax, event_time, event_duration)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Active power [pu on system base]")
    ax.set_title("WT active-power response")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_wt_power_sys_pu.png")


def plot_uic_pq_comparison(df_no_droop, df_droop, output_dir, event_time, event_duration) -> None:
    for col in ("P_uic_bus_sys_pu", "Q_uic_bus_sys_pu"):
        ensure_column(df_no_droop, col, "no-droop CSV")
        ensure_column(df_droop, col, "droop CSV")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], df_no_droop["P_uic_bus_sys_pu"], label="UIC P, without droop")
    ax.plot(df_no_droop["t"], df_no_droop["Q_uic_bus_sys_pu"], label="UIC Q, without droop")
    ax.plot(df_droop["t"], df_droop["P_uic_bus_sys_pu"], label="UIC P, with droop")
    ax.plot(df_droop["t"], df_droop["Q_uic_bus_sys_pu"], label="UIC Q, with droop")
    add_event_markers(ax, event_time, event_duration)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [pu on system base]")
    ax.set_title("UIC active and reactive power")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_uic_bus_PQ_sys_pu.png")


def plot_sync_generator_pq_comparison(df_no_droop, df_droop, output_dir, event_time, event_duration) -> None:
    cols = ("P_sync_generators_total_sys_pu", "Q_sync_generators_total_sys_pu")
    for col in cols:
        ensure_column(df_no_droop, col, "no-droop CSV")
        ensure_column(df_droop, col, "droop CSV")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], df_no_droop[cols[0]], label="Sync generators P, without droop")
    ax.plot(df_no_droop["t"], df_no_droop[cols[1]], label="Sync generators Q, without droop")
    ax.plot(df_droop["t"], df_droop[cols[0]], label="Sync generators P, with droop")
    ax.plot(df_droop["t"], df_droop[cols[1]], label="Sync generators Q, with droop")
    add_event_markers(ax, event_time, event_duration)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [pu on system base]")
    ax.set_title("Aggregate synchronous-generator response")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_sync_generators_PQ_sys_pu.png")


def plot_rotor_speeds_comparison(df_no_droop, df_droop, output_dir, event_time, event_duration) -> None:
    for col in ("omega_m_pu", "omega_e_pu"):
        ensure_column(df_no_droop, col, "no-droop CSV")
        ensure_column(df_droop, col, "droop CSV")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], df_no_droop["omega_m_pu"], label="Mechanical speed, without droop")
    ax.plot(df_no_droop["t"], df_no_droop["omega_e_pu"], label="Electrical speed, without droop")
    ax.plot(df_droop["t"], df_droop["omega_m_pu"], label="Mechanical speed, with droop")
    ax.plot(df_droop["t"], df_droop["omega_e_pu"], label="Electrical speed, with droop")
    add_event_markers(ax, event_time, event_duration)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [pu]")
    ax.set_title("WT drivetrain speeds")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_wt_rotor_speeds_pu.png")


def plot_tracking_error_comparison(df_no_droop, df_droop, output_dir, event_time, event_duration) -> None:
    for col in ("P_ref_sys_pu", "P_uic_bus_sys_pu"):
        ensure_column(df_no_droop, col, "no-droop CSV")
        ensure_column(df_droop, col, "droop CSV")

    error_no_droop = df_no_droop["P_ref_sys_pu"] - df_no_droop["P_uic_bus_sys_pu"]
    error_droop = df_droop["P_ref_sys_pu"] - df_droop["P_uic_bus_sys_pu"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], error_no_droop, label="Without droop")
    ax.plot(df_droop["t"], error_droop, label="With droop")
    ax.axhline(0.0, linestyle="--", label="Zero")
    add_event_markers(ax, event_time, event_duration)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$P_{\mathrm{ref}} - P_{\mathrm{UIC}}$ [pu on system base]")
    ax.set_title("UIC active-power tracking error")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_wt_power_tracking_error_pu.png")


def plot_droop_delta(df_droop, output_dir, event_time, event_duration) -> None:
    if "P_droop_delta_uic_pu" not in df_droop.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_droop["t"], df_droop["P_droop_delta_uic_pu"], label="With droop")
    ax.axhline(0.0, linestyle="--", label="Without droop / zero")
    add_event_markers(ax, event_time, event_duration)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Droop contribution [pu on UIC base]")
    ax.set_title("WT droop contribution")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_wt_droop_delta_uic_pu.png")


def plot_rocof_comparison(
    df_no_droop,
    df_droop,
    output_dir,
    event_time,
    event_duration,
) -> None:
    """Create a numerical RoCoF overlay as a diagnostic figure."""
    for source_name, df in (
        ("no-droop CSV", df_no_droop),
        ("droop CSV", df_droop),
    ):
        ensure_column(df, "t", source_name)
        ensure_column(df, "f_grid_hz", source_name)

        time_values = df["t"].to_numpy(dtype=float)
        if len(time_values) < 3 or np.any(np.diff(time_values) <= 0.0):
            raise ValueError(
                f"Cannot calculate RoCoF for {source_name}: "
                "time samples must be strictly increasing."
            )

    rocof_no_droop = np.gradient(
        df_no_droop["f_grid_hz"].to_numpy(dtype=float),
        df_no_droop["t"].to_numpy(dtype=float),
    )
    rocof_droop = np.gradient(
        df_droop["f_grid_hz"].to_numpy(dtype=float),
        df_droop["t"].to_numpy(dtype=float),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df_no_droop["t"],
        rocof_no_droop,
        label="Without droop",
    )
    ax.plot(
        df_droop["t"],
        rocof_droop,
        label="With droop",
    )
    ax.axhline(0.0, linestyle="--", label="Zero")
    add_event_markers(ax, event_time, event_duration)
    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
        scilimits=(-10, 10),
    )
    ax.set_ylim(-0.25, 0.25)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RoCoF [Hz/s]")
    ax.set_title("Rate of change of frequency")
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, "comparison_rocof_hz_per_s.png")


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



# ---------------------------------------------------------------------
# Additional figures for the extended WT/UIC logging.
#
# These functions are additive. The original plots and their appearance
# are left unchanged above.
# ---------------------------------------------------------------------
def common_column(df_no_droop, df_droop, candidates):
    """Return the first candidate column available in both CSV files."""
    for column in candidates:
        if column in df_no_droop.columns and column in df_droop.columns:
            return column
    return None


def plot_extended_two_cases(
    df_no_droop,
    df_droop,
    candidates,
    ylabel,
    output_dir,
    filename,
    event_time,
    event_duration,
    title=None,
    zero_line=False,
    nominal_line=None,
):
    """
    Add one simple overlay figure for a newly logged quantity.

    The existing plot functions are not changed. This helper is only used
    for columns that were added in the full WT/UIC logging update.
    """
    column = common_column(df_no_droop, df_droop, candidates)

    if column is None:
        print(
            f"Skipping {filename}: none of these columns exist in both CSV files: "
            f"{', '.join(candidates)}"
        )
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_no_droop["t"], df_no_droop[column], label="Without droop")
    ax.plot(df_droop["t"], df_droop[column], label="With droop")

    if nominal_line is not None:
        ax.axhline(
            nominal_line,
            linestyle="--",
            label=f"Nominal value ({nominal_line:g})",
        )

    if zero_line:
        ax.axhline(0.0, linestyle="--", label="Zero")

    # Uses the original dotted blue event-marker function unchanged.
    add_event_markers(ax, event_time, event_duration)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, filename)


def plot_extended_multisignal(
    df_no_droop,
    df_droop,
    signals,
    ylabel,
    output_dir,
    filename,
    event_time,
    event_duration,
    title,
    zero_line=False,
    nominal_line=None,
):
    """
    Add an overlay plot with multiple extended signals.

    signals is a list of:
        ([preferred_column, fallback_column, ...], legend label)

    Only signals available in both result CSV files are included.
    """
    resolved = []

    for candidates, label in signals:
        column = common_column(df_no_droop, df_droop, candidates)

        if column is None:
            print(
                f"{filename}: skipping signal '{label}' because these columns "
                f"are unavailable in one or both CSV files: {', '.join(candidates)}"
            )
            continue

        resolved.append((column, label))

    if not resolved:
        print(f"Skipping {filename}: no extended signals available.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for column, label in resolved:
        ax.plot(
            df_no_droop["t"],
            df_no_droop[column],
            label=f"{label}, without droop",
        )
        ax.plot(
            df_droop["t"],
            df_droop[column],
            label=f"{label}, with droop",
        )

    if nominal_line is not None:
        ax.axhline(
            nominal_line,
            linestyle="--",
            label=f"Nominal value ({nominal_line:g})",
        )

    if zero_line:
        ax.axhline(0.0, linestyle="--", label="Zero")

    # Uses the original dotted blue event-marker function unchanged.
    add_event_markers(ax, event_time, event_duration)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    save_figure(fig, output_dir, filename)


def plot_uic_actual_reference_pq(
    df_no_droop,
    df_droop,
    output_dir,
    event_time,
    event_duration,
):
    """
    Add detailed UIC actual-versus-reference P/Q overlays.

    The original comparison_uic_bus_PQ_sys_pu.png is preserved. These are
    extra figures using the newly logged reference signals.
    """
    plot_extended_multisignal(
        df_no_droop,
        df_droop,
        [
            (
                ["P_uic_bus_actual_sys_pu", "P_uic_bus_sys_pu"],
                "UIC active power, actual",
            ),
            (
                ["P_uic_bus_ref_sys_pu"],
                "UIC active power, reference",
            ),
        ],
        "Active power [pu on system base]",
        output_dir,
        "comparison_uic_active_power_actual_reference_pu.png",
        event_time,
        event_duration,
        "UIC active power: actual and reference",
    )

    plot_extended_multisignal(
        df_no_droop,
        df_droop,
        [
            (
                ["Q_uic_bus_actual_sys_pu", "Q_uic_bus_sys_pu"],
                "UIC reactive power, actual",
            ),
            (
                ["Q_uic_bus_ref_sys_pu"],
                "UIC reactive power, reference",
            ),
        ],
        "Reactive power [pu on system base]",
        output_dir,
        "comparison_uic_reactive_power_actual_reference_pu.png",
        event_time,
        event_duration,
        "UIC reactive power: actual and reference",
    )

def print_frequency_metrics(df_no_droop, df_droop) -> None:
    for source_name, df in (("Without droop", df_no_droop), ("With droop", df_droop)):
        ensure_column(df, "f_grid_hz", source_name)
        min_idx = df["f_grid_hz"].idxmin()
        f_min = float(df.loc[min_idx, "f_grid_hz"])
        t_min = float(df.loc[min_idx, "t"])
        print(f"{source_name:16s}: f_min = {f_min:.6f} Hz at t = {t_min:.3f} s")

    improvement_mhz = (
        float(df_droop["f_grid_hz"].min())
        - float(df_no_droop["f_grid_hz"].min())
    ) * 1000.0
    print(f"Improvement in frequency nadir: {improvement_mhz:.3f} mHz")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create overlay plots for LEOGO cases without and with WT droop."
    )
    parser.add_argument(
        "--no-droop-input",
        type=Path,
        default=PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "WT1_LEOGO_frequency_5MW_no_droop_deLoaded_MEASUREMENTS_graduallyRising.csv",
    )
    parser.add_argument(
        "--droop-input",
        type=Path,
        default=PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "WT1_LEOGO_frequency_5MW_droop_deLoaded_MEASUREMENTS_graduallyRising.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "casestudies" / "dyn_sim" / "logs" / "wt" / "plots" / "thesis" / "LEOGO_frequency_comparison" / "5MWsmoothLoadStep",
    )
    parser.add_argument("--event-time", type=float, default=5.0)
    parser.add_argument("--event-duration", type=float, default=50.0)
    args = parser.parse_args()

    if not args.no_droop_input.is_file():
        raise FileNotFoundError(f"No-droop CSV not found: {args.no_droop_input}")
    if not args.droop_input.is_file():
        raise FileNotFoundError(f"Droop CSV not found: {args.droop_input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_no_droop = pd.read_csv(args.no_droop_input)
    df_droop = pd.read_csv(args.droop_input)

    for name, df in (("no-droop CSV", df_no_droop), ("droop CSV", df_droop)):
        ensure_column(df, "t", name)
        ensure_column(df, "f_grid_hz", name)

    df_no_droop = df_no_droop.copy()
    df_droop = df_droop.copy()
    df_no_droop["frequency_deviation_hz"] = df_no_droop["f_grid_hz"] - 50.0
    df_droop["frequency_deviation_hz"] = df_droop["f_grid_hz"] - 50.0

    df_no_droop = add_unwrapped_current_angle(df_no_droop)
    df_droop = add_unwrapped_current_angle(df_droop)

    # Dedicated frequency formatting avoids misleading y-axis offsets for
    # near-zero numerical deviations around 50 Hz.
    plot_grid_frequency_comparison(
        df_no_droop,
        df_droop,
        args.output_dir,
        args.event_time,
        args.event_duration,
    )
    plot_frequency_deviation_comparison(
        df_no_droop,
        df_droop,
        args.output_dir,
        args.event_time,
        args.event_duration,
    )
    plot_two_cases(
        df_no_droop, df_droop, "I_uic_pu", "Current magnitude [pu on UIC base]",
        args.output_dir, "comparison_uic_current_pu.png",
        args.event_time, args.event_duration,
        title="UIC current comparison",
    )
    plot_two_cases(
        df_no_droop, df_droop, "V_WTG1_LV_pu", "Voltage [pu]",
        args.output_dir, "comparison_wtg1_lv_voltage_pu.png",
        args.event_time, args.event_duration,
        title="WTG1 LV voltage comparison", nominal_line=1.0,
    )
    plot_two_cases(
        df_no_droop, df_droop, "wind_speed_mps", "Wind speed [m/s]",
        args.output_dir, "comparison_wind_speed_mps.png",
        args.event_time, args.event_duration,
        title="Wind-speed comparison",
    )
    plot_two_cases(
        df_no_droop, df_droop, "pitch_deg", "Pitch angle [deg]",
        args.output_dir, "comparison_wt_pitch_deg.png",
        args.event_time, args.event_duration,
        title="WT pitch-angle comparison",
    )

    plot_power_comparison(df_no_droop, df_droop, args.output_dir, args.event_time, args.event_duration)
    plot_uic_pq_comparison(df_no_droop, df_droop, args.output_dir, args.event_time, args.event_duration)
    plot_sync_generator_pq_comparison(df_no_droop, df_droop, args.output_dir, args.event_time, args.event_duration)
    plot_rotor_speeds_comparison(df_no_droop, df_droop, args.output_dir, args.event_time, args.event_duration)
    plot_tracking_error_comparison(df_no_droop, df_droop, args.output_dir, args.event_time, args.event_duration)
    plot_droop_delta(df_droop, args.output_dir, args.event_time, args.event_duration)

    # -----------------------------------------------------------------
    # Additional comparison figures from the extended WT/UIC logging.
    # The original figures above are retained unchanged.
    # -----------------------------------------------------------------
    plot_extended_two_cases(
        df_no_droop,
        df_droop,
        ["T_mpt_wt_pu"],
        "Mechanical torque [pu on WT base]",
        args.output_dir,
        "comparison_wt_T_mpt_pu.png",
        args.event_time,
        args.event_duration,
        title="MPT mechanical-torque comparison",
    )

    plot_extended_multisignal(
        df_no_droop,
        df_droop,
        [
            (["P_aero_sys_pu"], "Aerodynamic power"),
            (["P_e_sys_pu"], "WT electrical power"),
        ],
        "Power [pu on system base]",
        args.output_dir,
        "comparison_wt_aero_electrical_power_sys_pu.png",
        args.event_time,
        args.event_duration,
        "WT aerodynamic and electrical power",
    )

    plot_extended_multisignal(
        df_no_droop,
        df_droop,
        [
            (["P_mpt_available_sys_pu"], "Available MPT power"),
            (["P_ref_instant_sys_pu"], "Instantaneous WT reference"),
        ],
        "Power [pu on system base]",
        args.output_dir,
        "comparison_wt_available_power_sys_pu.png",
        args.event_time,
        args.event_duration,
        "Available WT power and instantaneous reference",
    )

    plot_extended_multisignal(
        df_no_droop,
        df_droop,
        [
            (["v_bus_pu", "V_WTG1_LV_pu"], "UIC terminal voltage"),
            (["vi_mag_pu"], "UIC internal voltage"),
        ],
        "Voltage [pu]",
        args.output_dir,
        "comparison_uic_terminal_internal_voltage_pu.png",
        args.event_time,
        args.event_duration,
        "UIC terminal and internal voltage",
        nominal_line=1.0,
    )

    plot_uic_actual_reference_pq(
        df_no_droop,
        df_droop,
        args.output_dir,
        args.event_time,
        args.event_duration,
    )

    plot_extended_two_cases(
        df_no_droop,
        df_droop,
        ["i_a_angle_unwrapped_deg", "i_a_angle_deg"],
        "Current angle [deg]",
        args.output_dir,
        "comparison_uic_current_angle_deg.png",
        args.event_time,
        args.event_duration,
        title="UIC current-angle comparison",
    )

    plot_extended_multisignal(
        df_no_droop,
        df_droop,
        [
            (["P_base_uic_pu"], "Base WT reference"),
            (["P_ref_uic_pu"], "Total WT reference"),
            (["P_available_uic_pu"], "Available WT power"),
        ],
        "Active power [pu on UIC base]",
        args.output_dir,
        "comparison_wt_droop_command_uic_pu.png",
        args.event_time,
        args.event_duration,
        "WT command and available power",
    )

    plot_rocof_comparison(
        df_no_droop,
        df_droop,
        args.output_dir,
        args.event_time,
        args.event_duration,
    )

    print_frequency_metrics(df_no_droop, df_droop)
    print(f"\nSaved comparison plots in:\n{args.output_dir}")


if __name__ == "__main__":
    main()
