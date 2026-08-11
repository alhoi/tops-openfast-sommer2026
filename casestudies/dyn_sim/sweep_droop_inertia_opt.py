r"""
Parameter-optimisation sweep for the droop + virtual-inertia frequency support,
on the realistic LEOGO gas-turbine-trip event (15.8 MW), de-loaded OpenFAST turbine.

Two resumable blocks (one subprocess per config, crash-safe --skip-existing):

  main   K_droop x K_inertia grid at a fixed fast filter (2 Hz) and fixed de-load,
         LEOGO 5.3 Hz artifact damped (--fix-leogo-xqt). Maps how the two gains
         trade nadir (droop) against RoCoF (inertia), and where the WT over-rates.

  lpf    frequency-LPF corner sweep at a nominal (droop, inertia), with the LEOGO
         5.3 Hz artifact LEFT IN, to expose the inertia-vs-mode-rejection trade-off.

Every run is Region 3, de-loaded to ~8 MW so a matched-reserve droop can act; the
peak WT power is recorded so over-rating shows up as a constraint, not a crash.

Usage:
  # inspect the plan + time estimate, run nothing
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_droop_inertia_opt.py --dry-run
  # quick validation (one short run)
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_droop_inertia_opt.py --smoke
  # the full study (resumable)
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_droop_inertia_opt.py --block all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVER = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_FMU_sim.py"
OUT_ROOT = PROJECT_ROOT / "results" / "em_interaction_sweep" / "droop_inertia_opt"

WALL_PER_SIMS = 3.5      # rough fast.fmu wall seconds per simulated second
T_END = 90.0

# Region-3, de-loaded to ~8 MW (11.5 MNm de-load, measured), and the 15.8 MW
# gas-turbine-trip load step. These are shared by every run.
BASE = ["--wt-pref-mw", "12.88", "--zmq-grid"]
DELOAD = ["--deload-nm", "1.15e7", "--support-max-nm", "1.5e7", "--support-start", "5"]
GT_TRIP = ["--load-step-mw", "15.8", "--event-time", "20",
           "--event-duration", "70", "--load-ramp-on-s", "0.5"]

# main grid
K_DROOP = [1e7, 2e7, 3e7, 4e7]        # Nm/Hz
K_INERTIA = [0.0, 2e7, 5e7, 8e7]      # Nm.s/Hz
MAIN_LPF = "2.0"                      # fast filter so the inertia can act

# lpf sub-study (nominal gains, artifact left in)
NOM_DROOP, NOM_INERTIA = "2e7", "5e7"
LPF_CORNERS = [0.3, 0.5, 1.0, 2.0]   # Hz


def _e(x: float) -> str:
    return format(x, ".0e").replace("+", "").replace("-", "m")


def build_jobs(blocks: set[str]) -> list[dict]:
    jobs: list[dict] = []
    if "main" in blocks:
        for kd in K_DROOP:
            for ki in K_INERTIA:
                support = ["--droop-nm-per-hz", f"{kd:.0f}",
                          "--inertia-nm-s-per-hz", f"{ki:.0f}",
                          "--freq-lpf-hz", MAIN_LPF, "--freq-lpf-order", "2"]
                jobs.append(dict(
                    block="main", t_end=T_END,
                    tag=f"opt_d{_e(kd)}_i{_e(ki)}",
                    args=BASE + ["--fix-leogo-xqt"] + DELOAD + support + GT_TRIP,
                ))
    if "lpf" in blocks:
        for lpf in LPF_CORNERS:
            support = ["--droop-nm-per-hz", NOM_DROOP,
                      "--inertia-nm-s-per-hz", NOM_INERTIA,
                      "--freq-lpf-hz", f"{lpf}", "--freq-lpf-order", "2"]
            # NO --fix-leogo-xqt here: leave the ~5.3 Hz mode in to be pumped.
            jobs.append(dict(
                block="lpf", t_end=T_END,
                tag=f"lpf_{str(lpf).replace('.', 'p')}",
                args=BASE + DELOAD + support + GT_TRIP,
            ))
    for j in jobs:
        j["out"] = OUT_ROOT / j["block"] / f"{j['tag']}.csv"
    return jobs


def run_job(job: dict, *, log_dir: Path) -> tuple[bool, float]:
    out_rel = job["out"].relative_to(PROJECT_ROOT)
    cmd = [sys.executable, str(DRIVER), "--t-end", f"{job['t_end']}",
           "--fmu", "fast", "--out", str(out_rel)] + job["args"]
    job["out"].parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_dir / f"{job['tag']}.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=log,
                              stderr=subprocess.STDOUT)
    return (proc.returncode == 0 and job["out"].exists()), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", choices=["main", "lpf", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        j = build_jobs({"main"})[0]
        j = dict(j, t_end=20.0, out=OUT_ROOT / "_smoke" / f"{j['tag']}.csv")
        print(f"Smoke: one short run {j['tag']} "
              f"(~{20 * WALL_PER_SIMS / 60:.1f} min) ...", flush=True)
        ok, wall = run_job(j, log_dir=OUT_ROOT / "_smoke" / "_logs")
        print(f"  {'ok' if ok else 'FAILED'}  ({wall:.0f}s)")
        return

    blocks = {"main", "lpf"} if args.block == "all" else {args.block}
    jobs = build_jobs(blocks)
    pending = [j for j in jobs if args.force or not j["out"].exists()]
    est = sum(j["t_end"] * WALL_PER_SIMS for j in pending) / 60.0

    print(f"Droop/inertia optimisation sweep: {len(jobs)} configs "
          f"({len(jobs) - len(pending)} done, {len(pending)} to run)")
    print(f"Estimated remaining wall time: {est:.0f} min (~{est/60:.1f} h)\n")
    for j in jobs:
        mark = "skip" if j not in pending else "RUN "
        print(f"  [{mark}] {j['block']:>4}  {j['tag']:<20} t_end={j['t_end']:.0f}s")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    log_dir = OUT_ROOT / "_logs"
    n_ok = n_fail = 0
    for i, j in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {j['block']} {j['tag']} ...", flush=True)
        try:
            ok, wall = run_job(j, log_dir=log_dir)
        except Exception as exc:
            print(f"    ERROR: {exc}"); n_fail += 1; continue
        if ok:
            n_ok += 1; print(f"    ok ({wall:.0f}s)")
        else:
            n_fail += 1; print(f"    FAILED (see {log_dir/(j['tag']+'.log')})")
    print(f"\nDone. {n_ok} ok, {n_fail} failed. Re-run to resume.")


if __name__ == "__main__":
    main()
