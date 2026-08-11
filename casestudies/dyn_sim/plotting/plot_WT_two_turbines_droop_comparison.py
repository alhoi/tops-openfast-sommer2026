"""
Compare the two-turbine finite-grid results WITH and WITHOUT frequency droop.

Reads the two CSVs produced by
    casestudies/dyn_sim/test_WT_two_turbines_sim.py
        WT_two_turbines_noDroop.csv
        WT_two_turbines_droop.csv
and produces:

  1. frequency_comparison.png
     Grid frequency for both cases, with the load-step window shaded and the
     frequency nadir of each case annotated. This is the headline result:
     droop lifts the nadir and speeds recovery.

  2. turbine_response_comparison.png
     Per-turbine response (WT1 and WT2) with vs without droop:
     UIC active power, rotor speed, blade pitch and the droop power
     contribution. Because WT2 carries a realistic (few-percent) manufacturing
     mismatch and sits slightly in WT1's wake, the two turbines respond
     differently even though they are nominally the same machine.

The CSVs are only read, never modified. Figures are written to a separate
output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# This file is in casestudies/dyn_sim/plotting, so parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"

DEFAULT_NODROOP_CSV = CSV_DIR / "WT_two_turbines_noDroop.csv"
DEFAULT_DROOP_CSV = CSV_DIR / "WT_two_turbines_droop.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "two_turbines_plots"
)

NODROOP_STYLE = dict(color="tab:red", linestyle="--", linewidth=1.6)
DROOP_STYLE = dict(color="tab:green", linestyle="-", linewidth=1.8)

WT1_NODROOP = dict(color="tab:blue", linestyle="--", linewidth=1.4)
WT1_DROOP = dict(color="tab:blue", linestyle="-", linewidth=1.8)
WT2_NODROOP = dict(color="tab:orange", linestyle="--", linewidth=1.4)
WT2_DROOP = dict(color="tab:orange", linestyle="-", linewidth=1.8)


def load_csv(path: Path) -> pd.DataFrame:
    """Read a result CSV and sort it by time."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find result CSV:\n  {path}\n\n"
            "Run test_WT_two_turbines_sim.py first, or pass explicit paths "
            "with --nodroop-csv / --droop-csv."
        )
    return pd.read_csv(path).sort_values("t").reset_index(drop=True)


def shade_event(ax, event_start: float, event_dur: float) -> None:
    """Shade the load-step window."""
    ax.axvspan(
        event_start, event_start + event_dur,
        color="grey", alpha=0.15, label="load step",
    )


def apply_xlim(ax, df: pd.DataFrame, event_start: float, pre: float, full: bool):
    """Zoom to a window starting shortly before the event, unless --full."""
    if full:
        ax.set_xlim(float(df["t"].min()), float(df["t"].max()))
    else:
        ax.set_xlim(max(0.0, event_start - pre), float(df["t"].max()))


def autoscale_y_to_view(ax, series) -> None:
    """Tighten the y-limits to only the data currently visible in x.

    ``series`` is a list of (t, y) pandas Series pairs. Without this,
    matplotlib keeps the y-range of the whole trace, which can make a signal
    that varies only slightly look like a flat line ("too zoomed out").
    """
    x0, x1 = ax.get_xlim()
    los, his = [], []
    for t, y in series:
        mask = (t >= x0) & (t <= x1)
        if mask.any():
            los.append(float(y[mask].min()))
            his.append(float(y[mask].max()))
    if not los or not his:
        return
    lo, hi = min(los), max(his)
    span = hi - lo
    pad = 0.08 * span if span > 1e-12 else max(abs(hi), 1e-6) * 0.01
    ax.set_ylim(lo - pad, hi + pad)


def annotate_nadir(ax, df: pd.DataFrame, col: str, style: dict, label: str):
    """Mark the minimum of a column."""
    i = df[col].idxmin()
    t_min = float(df["t"].iloc[i])
    y_min = float(df[col].iloc[i])
    ax.plot(t_min, y_min, marker="o", color=style["color"], markersize=6)
    ax.annotate(
        f"{label}: {y_min:.3f} Hz",
        xy=(t_min, y_min),
        xytext=(6, -12 if label == "no droop" else 8),
        textcoords="offset points",
        color=style["color"],
        fontsize=9,
    )


def plot_frequency(
    df_no: pd.DataFrame,
    df_dr: pd.DataFrame,
    output_dir: Path,
    event_start: float,
    event_dur: float,
    pre: float,
    full: bool,
    show: bool,
) -> None:
    """Grid frequency: no-droop vs droop, with nadir annotations."""
    fig, ax = plt.subplots(figsize=(10, 5))

    shade_event(ax, event_start, event_dur)
    ax.plot(df_no["t"], df_no["f_grid_hz"], label="no droop", **NODROOP_STYLE)
    ax.plot(df_dr["t"], df_dr["f_grid_hz"], label="droop", **DROOP_STYLE)

    annotate_nadir(ax, df_no, "f_grid_hz", NODROOP_STYLE, "no droop")
    annotate_nadir(ax, df_dr, "f_grid_hz", DROOP_STYLE, "droop")

    f_nom = 50.0
    ax.axhline(f_nom, color="black", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("System grid frequency [Hz]")
    ax.set_title(
        "System grid frequency (shared by BOTH turbines): "
        "with vs without droop"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    apply_xlim(ax, df_no, event_start, pre, full)
    autoscale_y_to_view(
        ax,
        [(df_no["t"], df_no["f_grid_hz"]), (df_dr["t"], df_dr["f_grid_hz"])],
    )

    fig.tight_layout()
    out = output_dir / "frequency_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    if not show:
        plt.close(fig)


def plot_turbine_response(
    df_no: pd.DataFrame,
    df_dr: pd.DataFrame,
    output_dir: Path,
    event_start: float,
    event_dur: float,
    pre: float,
    full: bool,
    show: bool,
) -> None:
    """Per-turbine WT1/WT2 response, with vs without droop."""
    # (column suffix, y label, panel title)
    panels = [
        ("P_uic_sys_pu", "Power [pu, sys base]", "UIC active power"),
        ("P_aero_sys_pu", "Power [pu, sys base]", "Aerodynamic power"),
        ("P_ref_sys_pu", "Power [pu, sys base]", "Power reference"),
        ("omega_m_pu", "Rotor speed [pu]", "Rotor (mechanical) speed"),
        ("pitch_deg", "Pitch [deg]", "Blade pitch angle"),
        ("P_droop_sys_pu", "Power [pu, sys base]", "Droop power contribution"),
    ]

    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 3 * len(panels)),
                             sharex=True)

    for row, (suffix, ylabel, title) in enumerate(panels):
        for col, (tag, no_style, dr_style) in enumerate(
            (("WT1", WT1_NODROOP, WT1_DROOP), ("WT2", WT2_NODROOP, WT2_DROOP))
        ):
            ax = axes[row, col]
            column = f"{tag}_{suffix}"
            shade_event(ax, event_start, event_dur)
            ax.plot(df_no["t"], df_no[column],
                    label=f"{tag} no droop", **no_style)
            ax.plot(df_dr["t"], df_dr[column],
                    label=f"{tag} droop", **dr_style)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{tag} - {title}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            apply_xlim(ax, df_no, event_start, pre, full)
            autoscale_y_to_view(
                ax,
                [(df_no["t"], df_no[column]), (df_dr["t"], df_dr[column])],
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(
        "Turbine response with vs without droop "
        "(WT2 = realistic mismatch + wake of WT1)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = output_dir / "turbine_response_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    if not show:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two-turbine finite-grid results with and without "
            "frequency droop."
        )
    )
    parser.add_argument("--nodroop-csv", type=Path, default=DEFAULT_NODROOP_CSV)
    parser.add_argument("--droop-csv", type=Path, default=DEFAULT_DROOP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-start", type=float, default=5.0,
                        help="Load-step start time [s] (for shading/zoom).")
    parser.add_argument("--event-duration", type=float, default=15.0,
                        help="Load-step duration [s] (for shading).")
    parser.add_argument("--pre", type=float, default=1.0,
                        help="Seconds shown before the event when zoomed.")
    parser.add_argument("--full", action="store_true",
                        help="Show the full time range instead of zooming.")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df_no = load_csv(args.nodroop_csv)
    df_dr = load_csv(args.droop_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    nadir_no = float(df_no["f_grid_hz"].min())
    nadir_dr = float(df_dr["f_grid_hz"].min())
    print("Frequency nadir [Hz]:")
    print(f"  no droop : {nadir_no:.4f}")
    print(f"  droop    : {nadir_dr:.4f}  "
          f"(raised by {nadir_dr - nadir_no:+.4f} Hz)")

    print(f"Writing figures to: {args.output_dir}")
    plot_frequency(
        df_no, df_dr, args.output_dir,
        args.event_start, args.event_duration, args.pre, args.full, args.show,
    )
    plot_turbine_response(
        df_no, df_dr, args.output_dir,
        args.event_start, args.event_duration, args.pre, args.full, args.show,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
