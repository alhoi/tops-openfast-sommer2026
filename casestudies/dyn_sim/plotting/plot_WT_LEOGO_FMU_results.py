"""
Plot the LEOGO + high-fidelity OpenFAST wind turbine FMU co-simulation.

This is the FMU counterpart of ``plot_WT_LEOGO_results.py`` and reads the CSV
produced by ``test_WT_LEOGO_FMU_sim.py``
(``results/WT1_LEOGO_FMU_results.csv``).

The FMU sim exposes a different signal set from the simplified analytic run:
the aero-servo-elastic states (rotor speed, generator speed, generator torque,
blade pitch, hub wind speed) come straight from OpenFAST, and there is a
``load_step_scale`` column that records the smooth load-step envelope.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


# This file is located in:
# casestudies/dyn_sim/plotting/plot_WT_LEOGO_FMU_results.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "casestudies"
    / "dyn_sim"
    / "results"
    / "WT1_LEOGO_FMU_results.csv"
)

# Separate output directory so the FMU figures do not overwrite the
# simplified-turbine LEOGO figures.
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "casestudies"
    / "dyn_sim"
    / "logs"
    / "wt"
    / "plots"
    / "thesis"
    / "WT_FMU_UIC_LEOGO"
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
        description="Create thesis-style plots from WT1_LEOGO_FMU_results.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to WT1_LEOGO_FMU_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG figures.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help=(
            "Optional sub-folder name created inside --output-dir. Use this to "
            "keep runs with different configurations apart, e.g. "
            "--label foreaft_TwFADOF1 or --label sidetoside_TwSSDOF1."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open all figures after saving.",
    )

    args = parser.parse_args()

    if args.label:
        args.output_dir = args.output_dir / args.label

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find results file:\n{args.input}\n\n"
            "Run test_WT_LEOGO_FMU_sim.py first."
        )

    df = pd.read_csv(args.input)

    if "t" not in df.columns:
        raise ValueError(
            "The CSV must contain a column named 't'. "
            f"Found: {list(df.columns)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- OpenFAST aero / rotor signals (from the FMU) --------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_wind_mps.png",
        "Hub wind speed [m/s]",
        [
            ("fmu_Wind1VelX", "Hub wind speed"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_rotor_speed_rpm.png",
        "Speed [rpm]",
        [
            ("fmu_RotSpeed", "Rotor speed"),
            ("fmu_GenSpeed", "Generator speed"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_gentorque.png",
        "Generator torque [kN·m]",
        [
            ("fmu_GenTq", "Generator torque"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_pitch_deg.png",
        "Blade pitch [deg]",
        [
            ("fmu_BldPitch1", "Blade 1 pitch"),
        ],
    )

    # Tower-top fore-aft acceleration (fore-aft tower mode, ~0.236 Hz).
    # Key signal for detecting resonant interaction with a sine load applied
    # at the tower fore-aft natural frequency.
    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_tower_foreaft_accel.png",
        "Tower-top fore-aft acceleration [m/s$^2$]",
        [
            ("fmu_YawBrTAxp", "Tower-top fore-aft acceleration"),
        ],
    )

    # Tower-top side-to-side acceleration (side-to-side tower mode, TwSSDOF1).
    # This is the mode a generator-torque disturbance can couple into, since
    # generator reaction torque rocks the nacelle laterally. Only available
    # when running fast_debug.fmu (fast.fmu does not expose YawBrTAyp).
    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_tower_sidetoside_accel.png",
        "Tower-top side-to-side acceleration [m/s$^2$]",
        [
            ("fmu_YawBrTAyp", "Tower-top side-to-side acceleration"),
        ],
    )

    # --- Electrical coupling state --------------------------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_omega_e_pu.png",
        "Electrical speed [pu]",
        [
            ("omega_e_pu", r"Electrical speed $\omega_e$"),
        ],
    )

    # --- Grid frequency (LEOGO synchronous generators) ------------------

    if "f_grid_hz" in df.columns:
        save_plot(
            df,
            args.output_dir,
            "leogo_fmu_grid_frequency_hz.png",
            "Grid frequency [Hz]",
            [
                ("f_grid_hz", "Grid frequency (COI)"),
            ],
        )

    # --- Turbine power at the grid interface ----------------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_wt_power_pu.png",
        "Power [pu on system base]",
        [
            ("P_e_sys_pu", "WT electrical power"),
            ("P_ref_sys_pu", "Power reference"),
        ],
    )

    # --- UIC terminal / internal voltage --------------------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_voltages_pu.png",
        "Voltage [pu]",
        [
            ("V_WTG1_LV_pu", "Busbar WTG1 LV terminal voltage"),
            ("V_uic_internal_pu", "UIC internal voltage"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_WTG1_LV_angle_deg.png",
        "Voltage angle [deg]",
        [
            ("angle_WTG1_LV_deg", "Busbar WTG1 LV angle"),
        ],
    )

    # --- UIC bus-side power ---------------------------------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_uic_bus_PQ_pu.png",
        "Power [pu on system base]",
        [
            ("P_uic_bus_sys_pu", "UIC active power P"),
            ("Q_uic_bus_sys_pu", "UIC reactive power Q"),
        ],
    )

    # --- UIC current ----------------------------------------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_current_mag_pu.png",
        "Current magnitude [pu on UIC base]",
        [
            ("I_uic_pu", "UIC armature current"),
        ],
    )

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_current_angle_deg.png",
        "Current angle [deg]",
        [
            ("I_uic_angle_deg", "UIC armature-current angle"),
        ],
    )

    # --- LEOGO synchronous generators (total P and Q) -------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_sync_generators_PQ_pu.png",
        "Power [pu on system base]",
        [
            ("P_sync_generators_total_sys_pu", "Total synchronous-generator P"),
            ("Q_sync_generators_total_sys_pu", "Total synchronous-generator Q"),
        ],
    )

    # --- Applied load-step envelope -------------------------------------

    save_plot(
        df,
        args.output_dir,
        "leogo_fmu_load_step_scale.png",
        "Load-step scale [-]",
        [
            ("load_step_scale", "Smooth load-step envelope"),
        ],
    )

    print(f"\nFigures saved in:\n{args.output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
