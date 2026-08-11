"""
Plot the two-turbine infinite-bus results.

Reads the CSV produced by
    casestudies/dyn_sim/test_WT_two_turbines_sim.py
and, for each signal, shows:
  * top panel   : WT1 and WT2 overlaid (absolute values), and
  * bottom panel: the difference WT1 - WT2.

Because the two turbines are (by default) identical, their absolute traces
almost overlap; the difference panel is what actually reveals the
turbine-to-turbine interaction. The fault window is shaded, and the view is
zoomed around the event by default so the transient is easy to read.

The CSV is only read, never modified. Figures are written to a separate
output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# This file is in casestudies/dyn_sim/plotting, so parents[3] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"

DEFAULT_CSV = CSV_DIR / "WT_two_turbines_infinite_bus_results.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "two_turbines_plots"
)

# (column suffix, y-axis label, plot title, output filename)
SIGNALS = [
    ("omega_m_pu", "Rotor speed [pu]", "Mechanical (rotor) speed", "omega_m.png"),
    ("omega_e_pu", "Generator speed [pu]", "Electrical (generator) speed", "omega_e.png"),
    ("omega_slip_pu", "Slip [pu]", "Drivetrain slip (omega_m - omega_e)", "omega_slip.png"),
    ("pitch_deg", "Pitch angle [deg]", "Blade pitch angle", "pitch.png"),
    ("P_aero_sys_pu", "Power [pu, sys base]", "Aerodynamic power", "P_aero.png"),
    ("P_e_sys_pu", "Power [pu, sys base]", "Electrical power (turbine)", "P_e.png"),
    ("P_ref_sys_pu", "Power [pu, sys base]", "Power reference", "P_ref.png"),
    ("V_term_pu", "Voltage [pu]", "Converter terminal voltage", "V_term.png"),
    ("P_uic_sys_pu", "Power [pu, sys base]", "UIC active power", "P_uic.png"),
    ("Q_uic_sys_pu", "Power [pu, sys base]", "UIC reactive power", "Q_uic.png"),
    ("I_uic_pu", "Current [pu]", "UIC current magnitude", "I_uic.png"),
]

WT1_COLOR = "tab:blue"
WT2_COLOR = "tab:orange"


def load_csv(path: Path) -> pd.DataFrame:
    """Read the result CSV and sort it by time."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find result CSV:\n  {path}\n\n"
            "Run test_WT_two_turbines_sim.py first, or pass an explicit path "
            "with --csv."
        )
    return pd.read_csv(path).sort_values("t").reset_index(drop=True)


def shade_fault(ax, event_start: float, event_dur: float) -> None:
    """Shade the fault window."""
    ax.axvspan(
        event_start, event_start + event_dur,
        color="grey", alpha=0.2, label="event",
    )


def baseline(df: pd.DataFrame, col: str, event_start: float) -> float:
    """Pre-event steady value of a column (last sample before the event)."""
    pre = df.loc[df["t"] < event_start, col]
    return float(pre.iloc[-1]) if len(pre) else float(df[col].iloc[0])


def settle_end(
    df: pd.DataFrame,
    cols: list[str],
    event_start: float,
    frac: float = 0.02,
    min_window: float = 0.4,
) -> float:
    """Estimate when the transient in `cols` has effectively settled.

    Returns the time of the last sample whose deviation from the pre-event
    baseline still exceeds `frac` of the peak deviation, plus a small margin.
    Each signal therefore gets a zoom window matched to its own timescale
    (fast electrical transients vs. slow speed swings).
    """
    t_last = event_start + min_window
    for col in cols:
        if col not in df.columns:
            continue
        dev = (df[col] - baseline(df, col, event_start)).abs()
        peak = dev.max()
        if peak <= 1e-12:
            continue
        mask = (dev > frac * peak) & (df["t"] >= event_start)
        if mask.any():
            t_last = max(t_last, float(df.loc[mask, "t"].max()))
    span = t_last - event_start
    return t_last + max(0.3, 0.15 * span)


def compute_xlim(
    df: pd.DataFrame,
    cols: list[str],
    event_start: float,
    pre: float,
    post: float | None,
    full: bool,
) -> tuple[float, float] | None:
    """Zoom window: fixed if `post` given, else auto from settling time."""
    if full:
        return None
    lo = max(float(df["t"].min()), event_start - pre)
    if post is not None:
        hi = min(float(df["t"].max()), event_start + post)
    else:
        hi = min(float(df["t"].max()), settle_end(df, cols, event_start))
    return (lo, hi)


def autoscale_y(ax, series_list, mask) -> None:
    """Scale the y-axis to the data visible in the zoom window only."""
    vals = np.concatenate([s[mask].to_numpy() for s in series_list if mask.any()])
    if vals.size == 0:
        return
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = 0.08 * (hi - lo) if hi > lo else max(0.01, abs(hi) * 0.05)
    ax.set_ylim(lo - pad, hi + pad)


def plot_signal(
    df: pd.DataFrame,
    suffix: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
    event_start: float,
    event_dur: float,
    pre: float,
    post: float | None,
    full: bool,
) -> None:
    """Top: absolute WT1 vs WT2. Bottom: deviation from pre-event baseline.

    The turbines run at different operating points, so their absolute traces
    are offset. The deviation panel removes that offset and puts both turbines
    on the same scale, so the response induced on the *undisturbed* turbine
    (the interaction) is directly visible. The window and y-scale adapt to the
    signal's own timescale so fine detail stays visible.
    """
    col1, col2 = f"WT1_{suffix}", f"WT2_{suffix}"
    if col1 not in df.columns or col2 not in df.columns:
        return

    base1 = baseline(df, col1, event_start)
    base2 = baseline(df, col2, event_start)
    xlim = compute_xlim(df, [col1, col2], event_start, pre, post, full)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6.5),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    dev1, dev2 = df[col1] - base1, df[col2] - base2

    # Top: absolute values.
    ax_top.plot(df["t"], df[col1], color=WT1_COLOR, label="WT1 (disturbed)")
    ax_top.plot(df["t"], df[col2], color=WT2_COLOR, label="WT2 (interaction)")
    shade_fault(ax_top, event_start, event_dur)
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(title)
    ax_top.grid(True)
    ax_top.legend(loc="upper right")

    # Bottom: deviation from each turbine's own pre-event baseline.
    ax_bot.plot(df["t"], dev1, color=WT1_COLOR, label="WT1")
    ax_bot.plot(df["t"], dev2, color=WT2_COLOR, label="WT2")
    ax_bot.axhline(0.0, color="black", linewidth=0.8)
    shade_fault(ax_bot, event_start, event_dur)
    ax_bot.set_ylabel("deviation from\npre-event value")
    ax_bot.set_xlabel("Time [s]")
    ax_bot.grid(True)
    ax_bot.legend(loc="upper right")

    if xlim is not None:
        ax_bot.set_xlim(*xlim)
        mask = (df["t"] >= xlim[0]) & (df["t"] <= xlim[1])
        autoscale_y(ax_top, [df[col1], df[col2]], mask)
        autoscale_y(ax_bot, [dev1, dev2], mask)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_overview(
    df: pd.DataFrame,
    panels: list[tuple[str, str]],
    output_dir: Path,
    filename: str,
    supertitle: str,
    event_start: float,
    event_dur: float,
    pre: float,
    post: float | None,
    full: bool,
) -> None:
    """Multi-panel overview (deviation view) sharing one auto-zoom window."""
    cols = [f"WT1_{s}" for s, _ in panels] + [f"WT2_{s}" for s, _ in panels]
    xlim = compute_xlim(df, cols, event_start, pre, post, full)
    mask = None
    if xlim is not None:
        mask = (df["t"] >= xlim[0]) & (df["t"] <= xlim[1])

    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(9, 2.4 * len(panels)))
    for ax, (suffix, ylabel) in zip(axes, panels):
        col1, col2 = f"WT1_{suffix}", f"WT2_{suffix}"
        dev1 = df[col1] - baseline(df, col1, event_start)
        dev2 = df[col2] - baseline(df, col2, event_start)
        ax.plot(df["t"], dev1, color=WT1_COLOR, label="WT1 (disturbed)")
        ax.plot(df["t"], dev2, color=WT2_COLOR, label="WT2 (interaction)")
        ax.axhline(0.0, color="black", linewidth=0.8)
        shade_fault(ax, event_start, event_dur)
        ax.set_ylabel("Δ " + ylabel)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)
        if mask is not None:
            autoscale_y(ax, [dev1, dev2], mask)
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(supertitle)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


ELECTRICAL_PANELS = [
    ("P_uic_sys_pu", "UIC power [pu]"),
    ("Q_uic_sys_pu", "UIC reactive [pu]"),
    ("I_uic_pu", "UIC current [pu]"),
    ("V_term_pu", "Term. voltage [pu]"),
]

MECHANICAL_PANELS = [
    ("omega_e_pu", "Gen. speed [pu]"),
    ("omega_m_pu", "Rotor speed [pu]"),
    ("pitch_deg", "Pitch [deg]"),
    ("P_aero_sys_pu", "Aero power [pu]"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the two-turbine infinite-bus results (WT1 vs WT2)."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--event-start", type=float, default=5.0,
        help="Event start time [s] (for shading and zoom window).",
    )
    parser.add_argument(
        "--event-duration", type=float, default=0.5,
        help="Event duration [s] (for shading).",
    )
    parser.add_argument(
        "--pre", type=float, default=0.5,
        help="Seconds shown before the event in the zoomed view.",
    )
    parser.add_argument(
        "--post", type=float, default=None,
        help="Seconds shown after the event. Default: auto per signal.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Plot the full run instead of zooming around the event.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = load_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for suffix, ylabel, title, filename in SIGNALS:
        plot_signal(
            df, suffix, ylabel, title, filename, args.output_dir,
            args.event_start, args.event_duration, args.pre, args.post, args.full,
        )

    plot_overview(
        df, ELECTRICAL_PANELS, args.output_dir, "overview_electrical.png",
        "Electrical interaction (deviation from pre-event value)",
        args.event_start, args.event_duration, args.pre, args.post, args.full,
    )
    plot_overview(
        df, MECHANICAL_PANELS, args.output_dir, "overview_mechanical.png",
        "Mechanical interaction (deviation from pre-event value)",
        args.event_start, args.event_duration, args.pre, args.post, args.full,
    )

    print(f"Figures written to: {args.output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
