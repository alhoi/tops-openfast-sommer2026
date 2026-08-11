from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


# This file is located in:
# casestudies/dyn_sim/plotting/plot_WT_LEOGO_results.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "casestudies"
    / "dyn_sim"
    / "results"
    / "WT1_LEOGO_results.csv"
)

# Separate LEOGO output directory so the baseline figures are preserved.
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "casestudies"
    / "dyn_sim"
    / "logs"
    / "wt"
    / "plots"
    / "thesis"
    / "LEOGO"
)


def save_plot(df, output_dir, filename, y_label, signals):
    """
    Create one figure.

    signals:
        List of tuples: (csv_column_name, legend_label)
    """
    available = [
        (column, label)
        for column, label in signals
        if column in df.columns
    ]

    if not available:
        print(f"Skipping {filename}: required CSV columns not found.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for column, label in available:
        ax.plot(df["t"], df[column], label=label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True)

    if len(available) > 1:
        ax.legend()

    fig.tight_layout()

    output_file = output_dir / filename
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create thesis-style plots from WT1_LEOGO_results.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to WT1_LEOGO_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open all figures after saving.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find results file:\n{args.input}\n\n"
            "Run test_WT_LEOGO_sim.py first."
        )

    df = pd.read_csv(args.input)

    if "t" not in df.columns:
        raise ValueError(
            "The CSV must contain a column named 't'. "
            f"Found: {list(df.columns)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Wind turbine input and mechanical states
    save_plot(
        df,
        args.output_dir,
        "leogo_wind_mps.png",
        "Wind speed [m/s]",
        [
            ("wind_speed_mps", "Wind speed"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_speeds_pu.png",
        "Speed [pu]",
        [
            ("omega_m_pu", r"Mechanical speed $\omega_m$"),
            ("omega_e_pu", r"Electrical speed $\omega_e$"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_pitch_deg.png",
        "Pitch angle [deg]",
        [
            ("pitch_deg", "Pitch angle"),
        ],
    )

    # This plot is generated only after T_mpt_wt_pu is added to the simulation CSV.
    save_plot(
        df,
        args.output_dir,
        "leogo_T_mpt_pu.png",
        "MPT torque [pu]",
        [
            ("T_mpt_wt_pu", "MPT torque"),
        ],
    )

    # WT aerodynamic, electrical, and reference power
    save_plot(
        df,
        args.output_dir,
        "leogo_wt_power_pu.png",
        "Power [pu on system base]",
        [
            ("P_aero_sys_pu", "Aerodynamic power"),
            ("P_e_sys_pu", "WT electrical power"),
            ("P_ref_sys_pu", "Power reference"),
        ],
    )

    # UIC terminal and internal voltage
    save_plot(
        df,
        args.output_dir,
        "leogo_voltages_pu.png",
        "Voltage [pu]",
        [
            ("V_WTG1_LV_pu", "Busbar WTG1 LV terminal voltage"),
            ("V_uic_internal_pu", "UIC internal voltage"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_WTG1_LV_angle_deg.png",
        "Voltage angle [deg]",
        [
            ("angle_WTG1_LV_deg", "Busbar WTG1 LV angle"),
        ],
    )

    # UIC bus-side power
    save_plot(
        df,
        args.output_dir,
        "leogo_uic_bus_PQ_pu.png",
        "Power [pu on system base]",
        [
            ("P_uic_bus_sys_pu", "UIC active power P"),
            ("Q_uic_bus_sys_pu", "UIC reactive power Q"),
        ],
    )

    # UIC current
    save_plot(
        df,
        args.output_dir,
        "leogo_current_mag_pu.png",
        "Current magnitude [pu on UIC base]",
        [
            ("I_uic_pu", "UIC armature current"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_current_angle_deg.png",
        "Current angle [deg]",
        [
            ("I_uic_angle_deg", "UIC armature-current angle"),
        ],
    )

    # The LEOGO model has three synchronous generators.
    # The simulation stores their total P and Q.
    save_plot(
        df,
        args.output_dir,
        "leogo_sync_generators_PQ_pu.png",
        "Power [pu on system base]",
        [
            ("P_sync_generators_total_sys_pu", "Total synchronous-generator P"),
            ("Q_sync_generators_total_sys_pu", "Total synchronous-generator Q"),
        ],
    )

    print(f"\nFigures saved in:\n{args.output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()