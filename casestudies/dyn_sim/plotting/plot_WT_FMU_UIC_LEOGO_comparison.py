"""
Overlay the high-fidelity OpenFAST FMU turbine against the simplified analytic
WindTurbine on the LEOGO network, one figure per signal.

Both LEOGO runs share the same electrical interface (UIC_sig at Busbar WTG1 LV)
and the same load-step disturbance, so their grid-side signals can be compared
directly. This script reads the two result CSVs and never modifies them:

  FMU turbine        : results/WT1_LEOGO_FMU_results.csv   (test_WT_LEOGO_FMU_sim.py)
  Simplified turbine : results/WT1_LEOGO_results.csv       (test_WT_LEOGO_sim.py)

Run both sims with the SAME scenario first, e.g.

  python casestudies/dyn_sim/test_WT_LEOGO_FMU_sim.py --t-end 100 --event-time 5 --event-duration 50 --load-step-mw 5
  python casestudies/dyn_sim/test_WT_LEOGO_sim.py     --t-end 100 --event-time 5 --event-duration 50 --load-step-mw 5

Figures are written to logs/wt/plots/thesis/WT_FMU_UIC_LEOGO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# casestudies/dyn_sim/plotting/plot_WT_FMU_UIC_LEOGO_comparison.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results"

DEFAULT_FMU_CSV = RESULTS_DIR / "WT1_LEOGO_FMU_results.csv"
DEFAULT_SIMPLIFIED_CSV = RESULTS_DIR / "WT1_LEOGO_results.csv"

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

FMU_LABEL = "OpenFAST FMU turbine"
SIMPLIFIED_LABEL = "Simplified turbine"


def add_event_markers(ax, event_time, event_duration):
    """Mark load application and removal without touching the data."""
    ax.axvline(event_time, linestyle=":", linewidth=1.5, color="0.4")
    ax.axvline(
        event_time + event_duration, linestyle=":", linewidth=1.5, color="0.4"
    )


def overlay_plot(
    fmu,
    simplified,
    output_dir,
    filename,
    title,
    y_label,
    event_time,
    event_duration,
    t_min,
    t_max,
    signals,
):
    """
    Create one comparison figure.

    signals:
        List of (fmu_column, simplified_column, base_label, linestyle) tuples.
        Either column may be None if that model does not log the signal.
    """
    lines = []
    for fmu_col, simp_col, base_label, linestyle in signals:
        if fmu_col is not None and fmu_col in fmu.columns:
            lines.append((fmu["t"], fmu[fmu_col], f"{FMU_LABEL}{base_label}", linestyle))
        if simp_col is not None and simp_col in simplified.columns:
            lines.append(
                (simplified["t"], simplified[simp_col], f"{SIMPLIFIED_LABEL}{base_label}", linestyle)
            )

    if not lines:
        print(f"Skipping {filename}: no matching columns found.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for t, y, label, linestyle in lines:
        ax.plot(t, y, label=label, linestyle=linestyle)

    add_event_markers(ax, event_time, event_duration)
    if t_min is not None or t_max is not None:
        ax.set_xlim(left=t_min, right=t_max)

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay FMU vs simplified WT results on the LEOGO network."
    )
    parser.add_argument("--fmu-csv", type=Path, default=DEFAULT_FMU_CSV)
    parser.add_argument("--simplified-csv", type=Path, default=DEFAULT_SIMPLIFIED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-time", type=float, default=5.0)
    parser.add_argument("--event-duration", type=float, default=50.0)
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    fmu = pd.read_csv(args.fmu_csv).sort_values("t").reset_index(drop=True)
    simplified = (
        pd.read_csv(args.simplified_csv).sort_values("t").reset_index(drop=True)
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        fmu=fmu,
        simplified=simplified,
        output_dir=output_dir,
        event_time=args.event_time,
        event_duration=args.event_duration,
        t_min=args.t_min,
        t_max=args.t_max,
    )

    # 01 UIC active power (actual + reference)
    overlay_plot(
        **common,
        filename="01_uic_active_power_comparison.png",
        title="UIC active power at Busbar WTG1 LV",
        y_label="Active power [pu on system base]",
        signals=[
            ("P_uic_bus_sys_pu", "P_uic_bus_sys_pu", " $P_{UIC,bus}$", "-"),
            ("P_ref_sys_pu", "P_ref_sys_pu", " $P_{ref}$", "--"),
        ],
    )

    # 02 UIC reactive power
    overlay_plot(
        **common,
        filename="02_uic_reactive_power_comparison.png",
        title="UIC reactive power at Busbar WTG1 LV",
        y_label="Reactive power [pu on system base]",
        signals=[("Q_uic_bus_sys_pu", "Q_uic_bus_sys_pu", "", "-")],
    )

    # 03 WT electrical (torque-power) input
    overlay_plot(
        **common,
        filename="03_wt_electrical_power_comparison.png",
        title="WT electrical power input $P_e$",
        y_label="Active power [pu on system base]",
        signals=[("P_e_sys_pu", "P_e_sys_pu", "", "-")],
    )

    # 04 Electrical-side drivetrain speed
    overlay_plot(
        **common,
        filename="04_drivetrain_speed_comparison.png",
        title="Electrical-coupling drivetrain speed $\\omega_e$",
        y_label="Speed [pu]",
        signals=[("omega_e_pu", "omega_e_pu", "", "-")],
    )

    # 05 UIC terminal and internal voltage
    overlay_plot(
        **common,
        filename="05_uic_voltage_comparison.png",
        title="UIC terminal and internal-voltage magnitudes",
        y_label="Voltage magnitude [pu]",
        signals=[
            ("V_WTG1_LV_pu", "V_WTG1_LV_pu", " $|V_t|$", "-"),
            ("V_uic_internal_pu", "V_uic_internal_pu", " $|v_i|$", "--"),
        ],
    )

    # 06 UIC current magnitude
    overlay_plot(
        **common,
        filename="06_uic_current_magnitude_comparison.png",
        title="UIC current magnitude",
        y_label="Current magnitude [pu on UIC base]",
        signals=[("I_uic_pu", "I_uic_pu", "", "-")],
    )

    # 07 UIC current angle
    overlay_plot(
        **common,
        filename="07_uic_current_angle_comparison.png",
        title="UIC current angle",
        y_label="Current angle [deg]",
        signals=[("I_uic_angle_deg", "I_uic_angle_deg", "", "-")],
    )

    # 08 Terminal-bus voltage angle
    overlay_plot(
        **common,
        filename="08_bus_angle_comparison.png",
        title="Voltage angle at Busbar WTG1 LV",
        y_label="Angle [deg]",
        signals=[("angle_WTG1_LV_deg", "angle_WTG1_LV_deg", "", "-")],
    )

    # 09 Remaining synchronous generation, active
    overlay_plot(
        **common,
        filename="09_sync_generators_active_power_comparison.png",
        title="Total LEOGO synchronous generation, active power",
        y_label="Active power [pu on system base]",
        signals=[
            ("P_sync_generators_total_sys_pu", "P_sync_generators_total_sys_pu", "", "-")
        ],
    )

    # 10 Remaining synchronous generation, reactive
    overlay_plot(
        **common,
        filename="10_sync_generators_reactive_power_comparison.png",
        title="Total LEOGO synchronous generation, reactive power",
        y_label="Reactive power [pu on system base]",
        signals=[
            ("Q_sync_generators_total_sys_pu", "Q_sync_generators_total_sys_pu", "", "-")
        ],
    )

    # 11 Blade pitch (FMU: BldPitch1, simplified: pitch_deg)
    overlay_plot(
        **common,
        filename="11_pitch_comparison.png",
        title="Blade pitch angle",
        y_label="Pitch [deg]",
        signals=[("fmu_BldPitch1", "pitch_deg", "", "-")],
    )

    # 12 Hub wind speed (FMU: Wind1VelX, simplified: wind_speed_mps)
    overlay_plot(
        **common,
        filename="12_wind_speed_comparison.png",
        title="Hub-height wind speed",
        y_label="Wind speed [m/s]",
        signals=[("fmu_Wind1VelX", "wind_speed_mps", "", "-")],
    )

    # 13 Applied load-step envelope (sanity check that scenarios match)
    overlay_plot(
        **common,
        filename="13_load_step_scale.png",
        title="Applied load-step envelope",
        y_label="Load fraction [-]",
        signals=[("load_step_scale", "load_step_scale", "", "-")],
    )

    print(f"Comparison figures written to: {output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
