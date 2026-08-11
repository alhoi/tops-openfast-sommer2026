r"""
Optimal tuning of the WT frequency-support droop gain for the LEOGO slug-flow
-> tower side-to-side (SS) resonance case.

------------------------------------------------------------------------------
The optimisation problem
------------------------------------------------------------------------------
Tuning the droop gain K_droop [pu/Hz] trades off two *conflicting* objectives,
both extracted from one two-phase slug-flow simulation (periodic pulsation on
the tower SS resonance, then a constant +hold_mw load step):

  * frequency support   -> the sustained grid-frequency dip |Df_hold| after the
                           load step SHRINKS as K grows  (good, decreasing in K)
  * structural loading  -> the tower SS acceleration amplitude A_ss GROWS as K
                           grows, because droop modulates the WT power at the
                           resonant frequency  (bad, increasing in K)

Because both metrics are *monotonic* in the single decision variable K, a plain
weighted-sum scalar cost is ill-posed here: its optimum is dictated entirely by
the (arbitrary) weight, and for any physically neutral weighting it collapses to
the degenerate K = 0 (never worth adding droop on resonance -- see the printed
net-benefit diagnostic). The well-posed engineering formulation is therefore a
*constrained* one:

        maximise   frequency support        (minimise the sustained dip)
        subject to A_ss(K) / A_ss(0) <= r_max        (a tower-load budget)

i.e. "use as much frequency-support droop as the tower structure can tolerate".
The user-set structural allowance r_max (the acceptable SS amplification factor,
a fatigue/load-envelope input) is the single meaningful design knob.

------------------------------------------------------------------------------
The algorithm: bisection on the binding constraint
------------------------------------------------------------------------------
The SS amplification r(K) = A_ss(K)/A_ss(0) increases monotonically with K, so
the constrained optimum lies exactly on the constraint boundary r(K*) = r_max.
Finding it is a 1-D monotonic root-find, for which **bisection** is the right
tool:

  * derivative-free  -- each simulation is an expensive black box (~1 min) and
                        no gradient is available;
  * robust to the small numerical noise in the amplitude metrics -- bisection
    only needs the *sign* of r(K) - r_max, never its magnitude, so it cannot be
    thrown off near a flat optimum the way parabolic/secant methods can;
  * guaranteed linear convergence -- the K bracket halves every iteration, so
    ~7 evaluations pin K* to ~1 % of the search range.

Bracketing handles the two edge cases automatically: if even K_max stays within
the tower budget the problem is *support-limited* and K* = K_max; if the budget
is already exceeded at K -> 0 then K* = 0 (droop not worthwhile).

Why not a fancier optimiser? For one smooth, monotonic decision variable,
Nelder-Mead, NSGA-II or Bayesian optimisation would be over-engineered: they
buy nothing over bisection here and cost many more of the expensive simulations.
The full trade-off (Pareto) curve is simply r(K) vs support(K), which this
script also samples and plots.

Usage
-----
  .\.venv\Scripts\python.exe casestudies\dyn_sim\optimize_droop_tuning.py
  .\.venv\Scripts\python.exe casestudies\dyn_sim\optimize_droop_tuning.py --target-amp 1.5
  .\.venv\Scripts\python.exe casestudies\dyn_sim\optimize_droop_tuning.py --show
"""

from __future__ import annotations

import argparse
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from casestudies.dyn_sim.test_WT_LEOGO_slugflow_ss_sim import run_case

CSV_DIR = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results" / "csv_files"
FIG_DIR = PROJECT_ROOT / "results" / "em_interaction_sweep"

# Fraction of the periodic phase skipped (initial settling) before the SS
# amplitude and the frequency baseline are measured.
_SETTLE_FRAC = 0.3


def _metrics(df: pd.DataFrame, args: Namespace) -> dict[str, float]:
    """Two conflicting objectives from one two-phase run.

    Frequency is referenced per run to the settled pre-step operating point
    (mean over the zero-mean-load tail of the periodic phase) so the start-up
    de-load offset is removed and only the slug-induced deviation is measured.
    """
    t = df["t"].to_numpy()
    onset = args.onset
    hold_start = args.hold_start
    t_end = float(t[-1])

    per_lo = onset + _SETTLE_FRAC * (hold_start - onset)
    per = (t >= per_lo) & (t < hold_start)
    hold_win = min(20.0, 0.7 * (t_end - hold_start))
    hold = t >= (t_end - hold_win)

    freq = df["grid_freq_hz"].to_numpy()
    base = float(np.mean(freq[per]))
    dev_mhz = (freq - base) * 1e3

    ss = df["ss_accel_mps2"].to_numpy()
    a_ss = 0.5 * float(np.ptp(ss[per]))          # SS acceleration amplitude
    dip = float(np.mean(dev_mhz[hold]))          # sustained hold-phase dip [mHz]
    return {"a_ss": a_ss, "dip_mhz": dip, "base_hz": base}


def _make_args(K: float, args: Namespace) -> Namespace:
    """Build the run_case argument namespace for one droop gain K."""
    return Namespace(
        forcing_freq_hz=args.forcing_freq_hz,
        amp_mw=args.amp_mw,
        wind=args.wind,
        t_end=args.t_end,
        dt=args.dt,
        onset=args.onset,
        hold_start=args.hold_start,
        hold_mw=args.hold_mw,
        headroom_pu=args.headroom_pu,
        wt_droop=(K > 0.0),
        droop_gain_pu_per_hz=float(K),
        out=None,
    )


class Evaluator:
    """Caches expensive simulation evaluations and logs every one to disk."""

    def __init__(self, args: Namespace, log_csv: Path) -> None:
        self.args = args
        self.log_csv = log_csv
        self.cache: dict[float, dict[str, float]] = {}
        self.rows: list[dict[str, float]] = []

    def __call__(self, K: float) -> dict[str, float]:
        key = round(float(K), 4)
        if key in self.cache:
            return self.cache[key]
        t0 = time.perf_counter()
        df = run_case(args=_make_args(key, self.args))
        m = _metrics(df, self.args)
        m["K"] = key
        m["wall_s"] = time.perf_counter() - t0
        self.cache[key] = m
        self.rows.append(m)
        pd.DataFrame(self.rows).sort_values("K").to_csv(self.log_csv, index=False)
        print(f"    K={key:6.3f} pu/Hz -> A_ss={m['a_ss']:.3e} m/s^2, "
              f"dip={m['dip_mhz']:6.2f} mHz   ({m['wall_s']:.0f} s)")
        return m


def bisect_constraint(ev: Evaluator, a_ss0: float, lo: float, hi: float,
                      r_max: float, tol_k: float, max_iter: int) -> float:
    """Largest K in [lo, hi] with A_ss(K)/A_ss(0) <= r_max (monotonic r).

    Returns hi if the constraint never binds (support-limited) and lo if it is
    already violated at lo.
    """
    r_lo = ev(lo)["a_ss"] / a_ss0
    r_hi = ev(hi)["a_ss"] / a_ss0
    if r_hi <= r_max:
        print(f"  constraint never binds on [{lo:g}, {hi:g}] "
              f"(r_max={r_max:g}); optimum is support-limited at K={hi:g}.")
        return hi
    if r_lo >= r_max:
        print(f"  constraint already violated at K={lo:g} "
              f"(r={r_lo:.2f} >= {r_max:g}); droop not worthwhile.")
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = ev(mid)["a_ss"] / a_ss0
        if r_mid < r_max:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol_k:
            break
    return 0.5 * (lo + hi)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Constrained bisection tuning of the WT droop gain.")
    p.add_argument("--target-amp", type=float, default=2.0,
                   help="Tower-load budget: max allowed SS amplification "
                        "A_ss(K)/A_ss(0). The single design knob.")
    p.add_argument("--k-max", type=float, default=6.0,
                   help="Upper bound of the droop-gain search [pu/Hz].")
    p.add_argument("--tol-k", type=float, default=0.15,
                   help="Convergence tolerance on K [pu/Hz].")
    p.add_argument("--max-iter", type=int, default=10)
    # Two-phase slug-flow forcing (shortened defaults for a fast optimisation;
    # verify the final K* with the full 200 s run afterwards).
    p.add_argument("--forcing-freq-hz", type=float, default=0.234)
    p.add_argument("--amp-mw", type=float, default=2.0)
    p.add_argument("--wind", type=float, default=10.0)
    p.add_argument("--t-end", type=float, default=100.0)
    p.add_argument("--onset", type=float, default=10.0)
    p.add_argument("--hold-start", type=float, default=75.0)
    p.add_argument("--hold-mw", type=float, default=1.0)
    p.add_argument("--headroom-pu", type=float, default=0.15)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--out", type=str,
                   default=str(FIG_DIR / "slugflow_ss_droop_opt.png"))
    p.add_argument("--log-csv", type=str,
                   default=str(CSV_DIR / "slugflow_ss_droop_opt_evals.csv"))
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Droop-gain tuning (constrained bisection), tower-load budget "
          f"r_max = {args.target_amp:g}x SS amplification")
    print(f"  slug {args.amp_mw:g} MW @ {args.forcing_freq_hz:g} Hz, "
          f"hold +{args.hold_mw:g} MW @ {args.hold_start:g} s, "
          f"t_end {args.t_end:g} s, headroom {args.headroom_pu:g} pu\n")

    t_start = time.perf_counter()
    ev = Evaluator(args, Path(args.log_csv))

    # Baseline operating point (droop OFF, same headroom -> shared op-point).
    print("  baseline (droop OFF):")
    m0 = ev(0.0)
    a_ss0 = m0["a_ss"]
    dip0 = m0["dip_mhz"]
    print(f"  -> A_ss(0)={a_ss0:.3e} m/s^2,  dip(0)={dip0:.2f} mHz\n")

    print("  bisection on the SS-amplification constraint:")
    k_star = bisect_constraint(ev, a_ss0, 0.0, args.k_max,
                               args.target_amp, args.tol_k, args.max_iter)
    m_star = ev(k_star)
    amp_star = m_star["a_ss"] / a_ss0
    support_star = 100.0 * (1.0 - abs(m_star["dip_mhz"]) / max(abs(dip0), 1e-12))

    print(f"\n  OPTIMUM  K* = {k_star:.3f} pu/Hz")
    print(f"    SS amplification   = {amp_star:.2f}x  (budget {args.target_amp:g}x)")
    print(f"    sustained dip      = {m_star['dip_mhz']:.2f} mHz "
          f"(-{support_star:.1f}% vs droop OFF)")

    # ----- net-benefit diagnostic (why a weighted sum is degenerate) ---------
    ev_sorted = sorted(ev.rows, key=lambda r: r["K"])
    K_arr = np.array([r["K"] for r in ev_sorted])
    ass_arr = np.array([r["a_ss"] for r in ev_sorted])
    dip_arr = np.array([abs(r["dip_mhz"]) for r in ev_sorted])
    support = 1.0 - dip_arr / max(abs(dip0), 1e-12)       # in [0, 1)
    penalty = ass_arr / a_ss0 - 1.0                        # >= 0, grows with K
    net_benefit = support - penalty                        # lambda = 1 scalar cost
    print("\n  net-benefit diagnostic  B(K) = support_gain - SS_penalty "
          "(lambda=1):")
    for K, b in zip(K_arr, net_benefit):
        print(f"    K={K:6.3f}: B={b:+.3f}")
    if net_benefit.max() <= net_benefit[np.argmin(K_arr)] + 1e-9:
        print("    -> B is maximised at K=0: on resonance the marginal SS cost "
              "exceeds the marginal support gain, so an un-weighted scalar cost "
              "would reject droop entirely. This is why the constrained "
              "formulation above is the right one.")

    # ----- figure ------------------------------------------------------------
    order = np.argsort(K_arr)
    Ks = K_arr[order]
    amp_curve = ass_arr[order] / a_ss0
    support_curve = 100.0 * support[order]

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    C_SUP, C_SS = "#1f5fb0", "#c0392b"

    ax.plot(Ks, support_curve, "o-", color=C_SUP, lw=1.7, ms=6,
            label="frekvensstøtte [% dyppreduksjon]")
    ax.set_xlabel("Droop-gain  K$_{droop}$  [pu/Hz]")
    ax.set_ylabel("Frekvensstøtte\n[% dyppreduksjon]", color=C_SUP)
    ax.tick_params(axis="y", labelcolor=C_SUP)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.plot(Ks, amp_curve, "s--", color=C_SS, lw=1.7, ms=6,
             label="SS-forsterkning [×]")
    ax2.set_ylabel("Tårn SS-amplitude\nforsterkning [×]", color=C_SS)
    ax2.tick_params(axis="y", labelcolor=C_SS)
    ax2.axhline(args.target_amp, color=C_SS, ls=":", lw=1.2,
                label=f"tårn-budsjett r$_{{max}}$={args.target_amp:g}×")
    ax2.axhline(1.0, color="k", lw=0.6, alpha=0.4)

    ax.axvline(k_star, color="#111111", ls="-", lw=1.4, alpha=0.8)
    ax.annotate(f"K* = {k_star:.2f} pu/Hz\n"
                f"støtte −{support_star:.0f}%, SS {amp_star:.2f}×",
                xy=(k_star, support_star), xytext=(0.42, 0.28),
                textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#111111", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#111111",
                          alpha=0.9))

    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8.5,
              framealpha=0.9)
    ax.set_title("Optimal droop-tuning: maksimer frekvensstøtte innenfor "
                 "tårn-lastbudsjett\n(begrenset biseksjon på "
                 "SS-forsterkningsgrensen)", fontsize=11)

    out = Path(args.out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  Lagret figur: {out}")
    print(f"  Eval-logg:    {args.log_csv}")
    print(f"  Total tid:    {time.perf_counter() - t_start:.0f} s "
          f"({len(ev.rows)} simuleringer)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
