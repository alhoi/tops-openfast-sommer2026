r"""
Tower side-to-side (SS) resonance curve with and without the support-path notch.

Same idea and style as plot_ss_fa_excitation.py, but adds a third curve that
shows the 0.233 Hz notch on the frequency-support input flattens the resonance
the support otherwise excites.

Reuses the existing full-matrix SS sweep for two baseline curves
(full_matrix/ss/ss_f*_off.csv and *_on.csv) and runs the missing legs so the
figure shows the same five cases as the GT-trip comparison, with matching
colours:

  no support              droop 0,   inertia 0            grey   (reused *_off)
  droop                   droop 2e7, inertia 0            blue   (*_dr)
  droop + notch           droop 2e7, inertia 0,  notch    orange (*_drN)
  droop + inertia         droop 2e7, inertia 5e6          green  (reused *_on)
  droop + inertia + notch droop 2e7, inertia 5e6, notch   purple (*_onN)

Needs ElastoDyn SS-only (TwSSDOF1=True, TwFADOF1=False) and fast_debug.fmu
(exposes fmu_YawBrTAyp). ~22 min per new point at t_end 400.

Usage:
  .\.venv\Scripts\python.exe casestudies\dyn_sim\ss_notch_resonance_curve.py --dry-run
  .\.venv\Scripts\python.exe casestudies\dyn_sim\ss_notch_resonance_curve.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\ss_notch_resonance_curve.py --plot-only
"""
from __future__ import annotations

import argparse
import re
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
SWEEP = PROJECT_ROOT / "results" / "em_interaction_sweep" / "full_matrix"
SS_DIR = SWEEP / "ss"
OUT = SWEEP / "ss_notch_resonance.png"

SS_HZ = 0.233
T_END = 400.0
SS_COL = "fmu_YawBrTAyp"
NOTCH_HZ = 0.233
NOTCH_Q = 8.0
# Support-torque clip (symmetric), kept active on every leg so the controller
# can never over-rate. In this small sine sweep it never binds (resonance-
# driving torque oscillation ~600 Nm << 3e6), but it matches the GT-trip figure
# intent of no over-rating.
SUPPORT_MAX_NM = 3.0e6

# Notch leg swept at the SAME gains as the existing "on" curve (droop 2e7,
# inertia 5e6). The main SS_FREQS set (matches the *_off.csv resolution); the
# notch curve is expected flat, so fine near-peak resolution is not needed.
NOTCH_FREQS = [0.16, 0.19, 0.21, 0.223, 0.229, 0.235, 0.245, 0.26, 0.30]

# Legs this script RUNS (off/on curves are reused from the existing sweep).
# (suffix, droop [Nm/Hz], inertia [Nm.s/Hz], notch_hz, notch_q)
# The droop+inertia legs use inertia 8e7 (like the GT-trip cap figure) so the
# derivative term dominates and the resonance is clearly larger than droop-only;
# at 5e6 the inertia term is negligible at 0.233 Hz and coincides with droop.
# The default 3e6 support clip stays active (no over-rating) but does not bind:
# the resonance-driving torque oscillation is ~600 Nm (<0.03% of the clip).
RUN_LEGS = [
    ("i8",  2e7, 8e7, 0.0,      2.0),      # droop + inertia 8e7     -> green  (KEY first)
    ("i8N", 2e7, 8e7, NOTCH_HZ, NOTCH_Q),  # droop + inertia + notch -> purple (KEY first)
    ("dr",  2e7, 0.0, 0.0,      2.0),      # droop only              -> blue   (resumes 6/9)
    ("drN", 2e7, 0.0, NOTCH_HZ, NOTCH_Q),  # droop only + notch      -> orange
]


def _tag(s: float, fmt: str = ".3f") -> str:
    return format(s, fmt).replace(".", "p").replace("+", "").replace("-", "m")


def lockin(t, sig, f, t_lo, t_hi) -> float:
    m = (t >= t_lo) & (t <= t_hi)
    tt, s = t[m], np.asarray(sig[m], float)
    if tt.size < 8:
        return float("nan")
    s = s - s.mean()
    w = 2.0 * np.pi * f
    c = np.trapezoid(s * np.cos(w * tt), tt)
    q = np.trapezoid(s * np.sin(w * tt), tt)
    span = tt[-1] - tt[0]
    return 2.0 * np.hypot(c, q) / span if span > 0 else float("nan")


def run_leg(f: float, suffix: str, droop: float, inertia: float,
            notch_hz: float, notch_q: float, *, force: bool) -> Path:
    out = SS_DIR / f"ss_f{_tag(f)}_{suffix}.csv"
    if out.exists() and not force:
        print(f"  [skip] {suffix} f={f:.3f} Hz (exists)")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(DRIVER),
        "--t-end", f"{T_END}", "--fmu", "debug",
        "--wt-pref-mw", "12.88", "--zmq-grid", "--fix-leogo-xqt",
        "--droop-nm-per-hz", f"{droop:.0f}",
        "--inertia-nm-s-per-hz", f"{inertia:.0f}",
        "--support-max-nm", f"{SUPPORT_MAX_NM:.0f}",
        "--support-notch-hz", f"{notch_hz}", "--support-notch-q", f"{notch_q}",
        "--load-step-mw", "3.0", "--event-time", "10",
        "--event-duration", "400", "--load-ramp-on-s", "3",
        "--load-sine-mean", "0.0", "--load-sine-amplitude", "1.0",
        "--load-sine-freq-hz", f"{f}",
        "--out", str(out.relative_to(PROJECT_ROOT)),
    ]
    print(f"  [run ] {suffix} f={f:.3f} Hz (droop={droop:.0e}, "
          f"inertia={inertia:.0e}, notch={notch_hz} Hz Q{notch_q:.0f}) ...",
          flush=True)
    log = SS_DIR / f"ss_f{_tag(f)}_{suffix}.log"
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=fh,
                              stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not out.exists():
        print(f"    FAILED (see {log})")
    return out


def curve(sup: str):
    """Settled SS accel vs drive frequency for the given support suffix."""
    pts = []
    for fp in sorted(SS_DIR.glob(f"ss_f*_{sup}.csv")):
        m = re.search(r"_f([0-9]+p[0-9]+)_", fp.stem)
        if not m:
            continue
        f = float(m.group(1).replace("p", "."))
        df = pd.read_csv(fp)
        if SS_COL not in df or "t" not in df:
            continue
        t = df["t"].to_numpy()
        a = lockin(t, df[SS_COL].to_numpy(), f, 0.6 * t.max(), t.max())
        pts.append((f, a))
    return sorted(pts)


def plot(out_path: Path = OUT) -> None:
    # (suffix, label, colour, marker, linestyle) - colours match the GT-trip
    # freq_support_5way_cap.png figure so the cases line up across the poster.
    series_def = [
        ("off", "no freq. support",         "#7f7f7f", "o", "--"),
        ("dr",  "droop",                    "#1f77b4", "o", "-"),
        ("drN", "droop + notch",            "#ff7f0e", "s", "-"),
        ("i8",  "droop + inertia",          "#2ca02c", "o", "-"),
        ("i8N", "droop + inertia + notch",  "#9467bd", "s", "-"),
    ]
    series = [(curve(suf), lab, col, mk, ls)
              for suf, lab, col, mk, ls in series_def]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for pts, label, colour, mk, ls in series:
        if not pts:
            continue
        lw = 1.3 if ls == "--" else 1.8
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                color=colour, lw=lw, ls=ls, label=label)

    ax.axvline(SS_HZ, color="k", ls=":", lw=0.8)
    ax.text(SS_HZ, ax.get_ylim()[1], f"  SS mode {SS_HZ} Hz",
            fontsize=8, va="top", ha="left")
    ax.set_xlabel("Process-load frequency [Hz]")
    ax.set_ylabel("Settled tower-top SS acceleration\nat drive frequency  [m/s$^2$]")
    ax.set_title("Frequency-support notch removes the tower SS resonance\n"
                 "(OpenFAST, Region 3, SS-only)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    for pts, label, *_ in series:
        if pts:
            pk = max(pts, key=lambda p: p[1])
            print(f"  {label:26s}: {len(pts)} pts, peak {pk[1]:.4f} @ {pk[0]:.3f} Hz")
        else:
            print(f"  {label:26s}: (no data yet)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--plot-out", type=str, default=None,
                    help="write the figure to this path instead of the default "
                         "(does not overwrite ss_notch_resonance.png).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        plot(Path(args.plot_out) if args.plot_out else OUT)
        return

    pending = [(suf, d, i, nh, nq, f)
               for (suf, d, i, nh, nq) in RUN_LEGS
               for f in NOTCH_FREQS
               if args.force or not (SS_DIR / f"ss_f{_tag(f)}_{suf}.csv").exists()]
    est = len(pending) * T_END * 3.5 / 60.0
    print(f"SS resonance legs: {len(RUN_LEGS)} legs x {len(NOTCH_FREQS)} freqs, "
          f"{len(pending)} points to run (~{est:.0f} min, fast_debug.fmu)")
    for suf, d, i, nh, nq in RUN_LEGS:
        n_have = sum((SS_DIR / f"ss_f{_tag(f)}_{suf}.csv").exists()
                     for f in NOTCH_FREQS)
        print(f"  leg {suf:4s} (droop={d:.0e} inertia={i:.0e} "
              f"notch={nh} Q{nq:.0f}): {n_have}/{len(NOTCH_FREQS)} done")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    for suf, d, i, nh, nq, f in pending:
        run_leg(f, suf, d, i, nh, nq, force=args.force)
    plot()


if __name__ == "__main__":
    main()
