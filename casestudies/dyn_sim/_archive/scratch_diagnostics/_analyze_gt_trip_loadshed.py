"""Analyze a LEOGO GT-trip + under-frequency load-shedding run.

Reads one or two CSVs produced by test_WT_LEOGO_tower_sim.py (frequency-support
OFF and, optionally, ON) and reproduces the LEOGO reference dynamic event
(Svendsen et al. 2023, IET Energy Syst. Integr.): a generator trip modelled as
a permanent power-balance step at the main gas-turbine busbar, followed ~200 ms
later by shedding ~9 MW of water-injection pumps. The figure shows the grid
frequency dip and partial recovery, the wind-turbine power response, and the
tower side-to-side acceleration.

Generation loss is represented as an equivalent load step at the main busbar
(net power balance), so all three synchronous gensets keep responding; a
genuine machine trip would also remove one unit's inertia and governor, giving
a slightly deeper dip. This approximation is documented in the report.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LEOGO system base [MVA] (Main Bus A slack, 100 MVA in LEOGO_ps.py). Used to
# convert the logged per-unit powers to MW for the figure.
SYS_MVA = 100.0


def _window_mean(t, y, t0, t1):
    """Mean of y over the time window [t0, t1]."""
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y[mask]))


def _summary(t, f, p_wt, event_time, settle_window):
    """Frequency and WT-power metrics for one run."""
    baseline = _window_mean(t, f, event_time - 5.0, event_time)
    post = t >= event_time
    nadir = float(np.min(f[post]))
    settled = _window_mean(t, f, t[-1] - settle_window, t[-1])
    p_pre = _window_mean(t, p_wt, event_time - 5.0, event_time)
    p_peak = float(np.max(p_wt[post]))
    p_settled = _window_mean(t, p_wt, t[-1] - settle_window, t[-1])
    return {
        "baseline_hz": baseline,
        "nadir_hz": nadir,
        "settled_hz": settled,
        "dip_nadir_hz": baseline - nadir,
        "dip_settled_hz": baseline - settled,
        "p_pre_mw": p_pre,
        "p_peak_mw": p_peak,
        "p_settled_mw": p_settled,
    }


def _load(csv_path):
    df = pd.read_csv(csv_path)
    t = df["t"].to_numpy()
    f = df["grid_freq_hz"].to_numpy()
    p_wt = df["P_uic_bus_sys_pu"].to_numpy() * SYS_MVA
    ss = df["ss_accel_mps2"].to_numpy()
    return t, f, p_wt, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-csv", required=True,
                    help="CSV with frequency support OFF.")
    ap.add_argument("--on-csv", default=None,
                    help="Optional CSV with frequency support ON (overlaid).")
    ap.add_argument("--event-time", type=float, default=10.0,
                    help="Generator-trip time [s].")
    ap.add_argument("--shed-delay", type=float, default=0.2,
                    help="Load-shedding delay after the trip [s].")
    ap.add_argument("--settle-window", type=float, default=10.0,
                    help="Window [s] at the end used for the settled values.")
    ap.add_argument("--out", default="results/em_interaction_sweep/"
                                      "gt_trip_loadshed/gt_trip_loadshed.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    t_off, f_off, p_off, ss_off = _load(args.off_csv)
    s_off = _summary(t_off, f_off, p_off, args.event_time, args.settle_window)

    has_on = args.on_csv is not None
    if has_on:
        t_on, f_on, p_on, ss_on = _load(args.on_csv)
        s_on = _summary(t_on, f_on, p_on, args.event_time, args.settle_window)

    t_trip = args.event_time
    t_shed = args.event_time + args.shed_delay

    # ---- text summary -------------------------------------------------
    print("\nGT-trip + load-shedding summary (LEOGO event)")
    print(f"  trip at t = {t_trip:.2f} s, shed 9 MW at t = {t_shed:.2f} s")

    def _print_run(tag, s):
        print(f"  [{tag}] baseline {s['baseline_hz']:.4f} Hz | "
              f"nadir {s['nadir_hz']:.4f} Hz (dip {s['dip_nadir_hz']*1e3:.1f} mHz) "
              f"| settled {s['settled_hz']:.4f} Hz "
              f"(dip {s['dip_settled_hz']*1e3:.1f} mHz)")
        print(f"        WT power: pre {s['p_pre_mw']:.3f} MW | "
              f"peak {s['p_peak_mw']:.3f} MW | settled {s['p_settled_mw']:.3f} MW")

    _print_run("support OFF", s_off)
    if has_on:
        _print_run("support ON ", s_on)
        print(f"  Freq-support effect: nadir dip "
              f"{s_off['dip_nadir_hz']*1e3:.1f} -> {s_on['dip_nadir_hz']*1e3:.1f} mHz | "
              f"settled dip {s_off['dip_settled_hz']*1e3:.1f} -> "
              f"{s_on['dip_settled_hz']*1e3:.1f} mHz")
        ss_pk_off = float(np.max(np.abs(ss_off[t_off >= t_trip])))
        ss_pk_on = float(np.max(np.abs(ss_on[t_on >= t_trip])))
        print(f"  Tower SS peak: {ss_pk_off:.4f} -> {ss_pk_on:.4f} m/s^2 "
              f"(x{ss_pk_on/ss_pk_off:.2f})")

    # ---- figure -------------------------------------------------------
    off_color = "0.45"
    on_color = "#c1272d"

    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.4), sharex=True)

    # (1) grid frequency
    ax = axes[0]
    ax.plot(t_off, f_off, color=off_color, lw=1.8,
            label="frekvensstøtte av")
    if has_on:
        ax.plot(t_on, f_on, color=on_color, lw=1.8, label="frekvensstøtte på")
    ax.set_ylabel("Nettfrekvens [Hz]")
    ax.set_title("LEOGO GT-utfall + lastutkobling (9 MW etter 200 ms) "
                 "\u2014 nett- og tårnrespons", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # (2) WT electrical power
    ax = axes[1]
    ax.plot(t_off, p_off, color=off_color, lw=1.8, label="frekvensstøtte av")
    if has_on:
        ax.plot(t_on, p_on, color=on_color, lw=1.8, label="frekvensstøtte på")
    ax.set_ylabel("Vindturbin\neffekt [MW]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # (3) tower side-to-side acceleration
    ax = axes[2]
    ax.plot(t_off, ss_off, color=off_color, lw=1.4, label="frekvensstøtte av")
    if has_on:
        ax.plot(t_on, ss_on, color=on_color, lw=1.4, label="frekvensstøtte på")
    ax.set_ylabel("Tårn side-til-side\nakselerasjon [m/s\u00b2]")
    ax.set_xlabel("Tid [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # event markers on every panel (the trip and the 200 ms shed are very
    # close together, so they are labelled once, with a small text offset).
    for ax in axes:
        ax.axvline(t_trip, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.axvline(t_shed, color="tab:blue", ls=":", lw=1.2, alpha=0.8)
    y_top = axes[0].get_ylim()[1]
    axes[0].annotate("GT-utfall", xy=(t_trip, y_top), xytext=(-2, -12),
                     textcoords="offset points", fontsize=8, color="k",
                     ha="right", va="top")
    axes[0].annotate("lastutkobling (200 ms)", xy=(t_shed, y_top),
                     xytext=(6, -12), textcoords="offset points", fontsize=8,
                     color="tab:blue", ha="left", va="top")

    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure written to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
