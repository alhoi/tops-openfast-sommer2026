r"""
Three-way frequency-support verification for the OpenFAST (FMU) turbine.

Applies one sharp, sustained load step (a GT-trip-like generation loss) at the
LEOGO main bus and compares three controller configurations:

  none            no frequency support (droop = 0, inertia = 0)
  droop           proportional droop only
  droop_inertia   droop + synthetic (virtual) inertia

The step response separates the two contributions:
  * synthetic inertia reacts to df/dt  -> lowers the RoCoF and shallows the nadir
    (a transient effect that vanishes once df/dt -> 0),
  * droop reacts to the frequency deviation -> holds the frequency up and shrinks
    the settled deviation.

For each case it computes the RoCoF, the frequency nadir and the settled
deviation, prints a comparison table, and plots the grid frequency and the
turbine electrical power for the three cases.

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\freq_support_3way.py            # run + plot
  .\.venv\Scripts\python.exe casestudies\dyn_sim\freq_support_3way.py --plot-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVER = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_FMU_sim.py"
OUT_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep" / "freq_support_3way"

T_END = 90.0
EVENT_TIME = 20.0          # s, load-step onset
# Realistic LEOGO single contingency (N-1): loss of one of the three 28 MVA
# gas turbines, each dispatched at 15.8 MW (LEOGO_ps.py). The two remaining
# units (15.8 -> ~23.7 MW each) cover the deficit within their 28 MVA rating,
# which is why this is the design event; the ~9 MW under-frequency load shed is
# the mitigation of exactly this trip.
LOAD_STEP_MW = 15.8        # sustained generation deficit = one gas-turbine trip

# (tag, droop [Nm/Hz], inertia [Nm.s/Hz], freq-LPF [Hz], torque clip [Nm], colour)
# The turbine is DE-LOADED (curtailed) so it holds a power reserve. At the event
# the frequency dip makes the droop unwind the de-load, releasing the reserve
# UPWARD toward the rating, never past it (a 15 MW machine cannot make more).
# The last case also opens the frequency filter (2 Hz) so the fast df/dt reaches
# the synthetic inertia.
CASES = [
    ("deload_none",          0.0, 0.0, 0.5, 1.5e7, "#d62728"),
    ("deload_droop",         2e7, 0.0, 0.5, 1.5e7, "#1f77b4"),
    ("deload_droop_inertia", 2e7, 5e7, 2.0, 1.5e7, "#2ca02c"),
]

DELOAD_NM = 1.15e7        # standing de-load: curtails ~14.5 MW to ~7.8 MW (measured),
                          # sized so passive + droop lands near the 15 MW rating
SUPPORT_START = 5.0       # de-load established well before the event at EVENT_TIME


def run_case(tag: str, droop: float, inertia: float, lpf: float,
             max_nm: float, *, force: bool, notch_hz: float = 0.0,
             notch_q: float = 2.0, suffix: str = "",
             max_over_nm: float | None = None, perfect_tracking: int = 0,
             fmu: str = "fast") -> Path:
    out = OUT_DIR / f"{tag}{suffix}.csv"
    if out.exists() and not force:
        print(f"  [skip] {tag}{suffix} (exists)")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(DRIVER),
        "--t-end", f"{T_END}", "--fmu", f"{fmu}",
        "--wt-pref-mw", "12.88", "--zmq-grid", "--fix-leogo-xqt",
        "--perfect-tracking", f"{perfect_tracking}",
        "--droop-nm-per-hz", f"{droop:.0f}",
        "--inertia-nm-s-per-hz", f"{inertia:.0f}",
        "--deload-nm", f"{DELOAD_NM:.0f}",
        "--freq-lpf-hz", f"{lpf}", "--freq-lpf-order", "2",
        "--support-notch-hz", f"{notch_hz}", "--support-notch-q", f"{notch_q}",
        "--support-start", f"{SUPPORT_START}", "--support-max-nm", f"{max_nm:.0f}",
        "--load-step-mw", f"{LOAD_STEP_MW}", "--event-time", f"{EVENT_TIME}",
        "--event-duration", f"{T_END - EVENT_TIME}", "--load-ramp-on-s", "0.5",
        "--zmq-log", str((OUT_DIR / f"{tag}{suffix}_zmq.csv").relative_to(PROJECT_ROOT)),
        "--out", str(out.relative_to(PROJECT_ROOT)),
    ]
    if max_over_nm is not None:
        cmd += ["--support-max-over-nm", f"{max_over_nm:.0f}"]
    print(f"  [run ] {tag}{suffix} (droop={droop:.0e}, inertia={inertia:.0e}, "
          f"lpf={lpf} Hz, notch={notch_hz} Hz, clip={max_nm:.0e}) ...", flush=True)
    log = OUT_DIR / f"{tag}{suffix}.log"
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=fh,
                              stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out.exists():
        print(f"    FAILED (see {log})")
    return out


def metrics(df: pd.DataFrame) -> dict[str, float]:
    """RoCoF, nadir and settled deviation from f_grid_hz, referenced to the
    pre-event frequency."""
    t = df["t"].to_numpy()
    f = df["f_grid_hz"].to_numpy()
    pre = f[(t >= EVENT_TIME - 5) & (t < EVENT_TIME)].mean()   # baseline

    # RoCoF: steepest slope in the first 2 s after the event (mHz/s).
    win = (t >= EVENT_TIME) & (t <= EVENT_TIME + 2.0)
    if win.sum() > 3:
        dfdt = np.gradient(f[win], t[win])
        rocof = float(np.max(np.abs(dfdt))) * 1000.0
    else:
        rocof = float("nan")

    post = t >= EVENT_TIME
    nadir_dev = float(np.min(f[post]) - pre) * 1000.0          # mHz
    settled = f[t >= T_END - 8.0].mean()
    settled_dev = float(settled - pre) * 1000.0                # mHz
    return {"pre_hz": pre, "rocof_mhz_s": rocof,
            "nadir_dev_mhz": nadir_dev, "settled_dev_mhz": settled_dev}


def analyze_and_plot(suffix: str = "", notch_hz: float = 0.0,
                     overlay: str | None = None,
                     overlay_label: str = "") -> None:
    frames = {}
    for tag, *_ in CASES:
        fp = OUT_DIR / f"{tag}{suffix}.csv"
        if fp.exists():
            frames[tag] = pd.read_csv(fp)

    if not frames:
        print("No result CSVs found; run without --plot-only first.")
        return

    print(f"\n{'case':>14}  {'RoCoF [mHz/s]':>13}  {'nadir [mHz]':>12}  "
          f"{'settled [mHz]':>13}")
    rows = {}
    for tag, *_ in CASES:
        if tag not in frames:
            continue
        m = metrics(frames[tag])
        rows[tag] = m
        print(f"{tag:>14}  {m['rocof_mhz_s']:>13.1f}  "
              f"{m['nadir_dev_mhz']:>12.1f}  {m['settled_dev_mhz']:>13.1f}")

    # Relative improvements vs the de-loaded no-support baseline.
    if "deload_none" in rows:
        b = rows["deload_none"]
        print("\nImprovement vs. de-loaded, no support:")
        for tag in ("deload_droop", "deload_droop_inertia"):
            if tag in rows:
                dr = 100 * (1 - abs(rows[tag]["rocof_mhz_s"]) / abs(b["rocof_mhz_s"]))
                dn = 100 * (1 - abs(rows[tag]["nadir_dev_mhz"]) / abs(b["nadir_dev_mhz"]))
                print(f"  {tag:>14}: RoCoF {dr:+.0f}%,  nadir {dn:+.0f}%")

    fig, (ax_f, ax_p) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for tag, _d, _i, _l, _m, colour in CASES:
        if tag not in frames:
            continue
        df = frames[tag]
        t = df["t"].to_numpy()
        ls = "--" if tag == "deload_none" else "-"
        ax_f.plot(t, df["f_grid_hz"], color=colour, lw=1.3, ls=ls, label=tag)
        if "P_e_sys_pu" in df:
            ax_p.plot(t, 100.0 * df["P_e_sys_pu"], color=colour, lw=1.3,
                      ls=ls, label=tag)
    if overlay:
        ofp = OUT_DIR / f"{overlay}.csv"
        if ofp.exists():
            odf = pd.read_csv(ofp)
            ot = odf["t"].to_numpy()
            lbl = overlay_label or overlay
            ax_f.plot(ot, odf["f_grid_hz"], color="#9467bd", lw=1.7, label=lbl)
            if "P_e_sys_pu" in odf:
                ax_p.plot(ot, 100.0 * odf["P_e_sys_pu"], color="#9467bd",
                          lw=1.7, label=lbl)
        else:
            print(f"  (overlay {ofp.name} not found)")
    ax_f.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
    ax_p.axvline(EVENT_TIME, color="k", ls=":", lw=0.8)
    ax_p.axhline(15.0, color="k", ls="--", lw=0.9)
    ax_p.text(60, 15.0, " 15 MW rating", va="bottom", ha="right",
              fontsize=8, color="k")
    ax_f.set_ylabel("Grid frequency  $f_{grid}$  [Hz]")
    ax_p.set_ylabel("WT electrical power  [MW]")
    ax_p.set_xlabel("Time [s]")
    notch_note = f"  (support notch {notch_hz:.3f} Hz)" if notch_hz > 0 else ""
    ax_f.set_title(f"De-loaded frequency support on a {LOAD_STEP_MW:.1f} MW "
                   f"gas-turbine trip (OpenFAST, Region 3){notch_note}")
    for ax in (ax_f, ax_p):
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_png = OUT_DIR / f"freq_support_3way{suffix}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--notch-hz", type=float, default=0.0,
                    help="Support-path notch centre frequency [Hz] (0 = off).")
    ap.add_argument("--notch-q", type=float, default=2.0,
                    help="Support-path notch quality factor.")
    ap.add_argument("--suffix", default="",
                    help="Suffix for the output CSVs and PNG (avoid overwriting).")
    ap.add_argument("--di-droop", type=float, default=None,
                    help="Override droop [Nm/Hz] of the droop+inertia case.")
    ap.add_argument("--di-inertia", type=float, default=None,
                    help="Override inertia [Nm.s/Hz] of the droop+inertia case.")
    ap.add_argument("--di-lpf", type=float, default=None,
                    help="Override freq-LPF corner [Hz] of the droop+inertia case.")
    ap.add_argument("--max-over-nm", type=float, default=None,
                    help="Asymmetric upper clip on the torque offset [Nm] to cap "
                         "the inertia burst near rating (applied to all cases).")
    ap.add_argument("--perfect-tracking", type=int, default=0, choices=[0, 1],
                    help="UIC perfect_tracking (0 = coupled, 1 = converter holds "
                         "nominal internal frequency, applied to all cases).")
    ap.add_argument("--fmu", default="fast", choices=["fast", "debug"],
                    help="Which OpenFAST FMU to use (debug exposes fmu_YawBrTAyp "
                         "= tower SS acceleration, but is much slower).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these case tags (e.g. deload_droop_inertia).")
    ap.add_argument("--overlay", default=None,
                    help="CSV stem (in the results dir) to overlay as a purple "
                         "line, e.g. deload_droop_inertia_wnotch.")
    ap.add_argument("--overlay-label", default="",
                    help="Legend label for the overlay line.")
    args = ap.parse_args()

    cases = [list(c) for c in CASES]
    if args.only:
        cases = [c for c in cases if c[0] in args.only]
    for c in cases:
        if c[0] == "deload_droop_inertia":
            if args.di_droop is not None:
                c[1] = args.di_droop
            if args.di_inertia is not None:
                c[2] = args.di_inertia
            if args.di_lpf is not None:
                c[3] = args.di_lpf

    if not args.plot_only:
        print("Three-way frequency-support runs "
              f"(~{len(cases) * T_END * 3.3 / 60:.0f} min):")
        for tag, droop, inertia, lpf, max_nm, _c in cases:
            run_case(tag, droop, inertia, lpf, max_nm, force=args.force,
                     notch_hz=args.notch_hz, notch_q=args.notch_q,
                     suffix=args.suffix, max_over_nm=args.max_over_nm,
                     perfect_tracking=args.perfect_tracking, fmu=args.fmu)
    if not args.only:
        analyze_and_plot(suffix=args.suffix, notch_hz=args.notch_hz,
                         overlay=args.overlay, overlay_label=args.overlay_label)


if __name__ == "__main__":
    main()
