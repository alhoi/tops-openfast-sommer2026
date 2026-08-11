r"""
Full electromechanical-interaction sweep matrix for the OpenFAST (FMU) turbine.

Runs resumable experiment blocks against ``test_WT_LEOGO_FMU_sim.py``, one
subprocess per operating point so the overnight run is crash-safe: re-running
the script skips points whose CSV already exists (resume).

Blocks
------
  ss         Tower side-to-side resonance curve. Sweeps a sustained sinusoidal
             process load around the SS mode (~0.229 Hz) and records the settled
             tower-top side-to-side acceleration. Needs fast_debug.fmu.
  torsion    Drivetrain torsion resonance curve. Sweeps the same process load
             around 3.05-3.49 Hz and records the shaft torque.
  stability  Droop-gain x frequency-LPF map. Does a high-gain droop reading the
             raw COI frequency pump the lightly damped ~5.3 Hz LEOGO q-axis mode
             into the generator torque? The LEOGO artifact is left IN (no
             --fix-leogo-xqt) so the mode is present to interact with.

Every run is Region 3 (--wt-pref-mw 12.88, --zmq-grid). Each block is run with
frequency support OFF and ON. The FMU wind speed is baked into OpenFAST, so a
Region-2 vs Region-3 comparison needs a separate FMU-wind change and is not part
of this unattended matrix.

Usage
-----
  # inspect the plan and time estimate, run nothing
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_em_full_matrix.py --dry-run
  # quick end-to-end validation (one short point per FMU variant)
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_em_full_matrix.py --smoke
  # the overnight run (resumable; re-run to resume after a crash)
  .\.venv\Scripts\python.exe casestudies\dyn_sim\sweep_em_full_matrix.py --block all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVER = PROJECT_ROOT / "casestudies" / "dyn_sim" / "test_WT_LEOGO_FMU_sim.py"
OUT_ROOT = PROJECT_ROOT / "results" / "em_interaction_sweep" / "full_matrix"

WALL_PER_SIMS = 3.3   # rough FMU wall-clock seconds per simulated second

# Region-3 operating point shared by every run.
BASE = ["--wt-pref-mw", "12.88", "--zmq-grid"]

# Frequency-support levels (OFF / full droop + synthetic inertia).
SUPPORT = {
    "off": ["--droop-nm-per-hz", "0", "--inertia-nm-s-per-hz", "0"],
    "on":  ["--droop-nm-per-hz", "2e7", "--inertia-nm-s-per-hz", "5e6"],
}

SS_FREQS = [0.16, 0.19, 0.21, 0.223, 0.229, 0.235, 0.245, 0.26, 0.30]
TORSION_FREQS = [2.6, 2.9, 3.05, 3.2, 3.35, 3.49, 3.7, 4.0]
STAB_DROOP = [0.0, 1e7, 3e7, 5e7, 8e7]     # Nm/Hz  (0 = support off baseline)
STAB_LPF = [0.0, 0.2, 0.5, 1.0]            # Hz     (0 = filter off)


def _tag(s: float, fmt: str) -> str:
    return format(s, fmt).replace(".", "p").replace("+", "").replace("-", "m")


def build_jobs(blocks: set[str]) -> list[dict]:
    """Return the list of sweep jobs (one per operating point)."""
    jobs: list[dict] = []

    if "ss" in blocks:
        for sup in ("off", "on"):
            for f in SS_FREQS:
                dist = [
                    "--load-step-mw", "3.0", "--event-time", "10",
                    "--event-duration", "400", "--load-ramp-on-s", "3",
                    "--load-sine-mean", "0.0", "--load-sine-amplitude", "1.0",
                    "--load-sine-freq-hz", f"{f}",
                ]
                jobs.append(dict(
                    block="ss", fmu="debug", t_end=400.0,
                    tag=f"ss_f{_tag(f, '.3f')}_{sup}",
                    args=BASE + ["--fix-leogo-xqt"] + SUPPORT[sup] + dist,
                ))

    if "fa" in blocks:
        # Tower fore-aft sweep, support OFF and ON. The off curve is the passive
        # floor (grid-forming UIC contribution), not exactly zero, so both are
        # kept for a symmetric figure. Needs ElastoDyn TwFADOF1=True,
        # TwSSDOF1=False; fast.fmu exposes YawBrTAxp.
        for sup in ("off", "on"):
            for f in SS_FREQS:
                dist = [
                    "--load-step-mw", "3.0", "--event-time", "10",
                    "--event-duration", "400", "--load-ramp-on-s", "3",
                    "--load-sine-mean", "0.0", "--load-sine-amplitude", "1.0",
                    "--load-sine-freq-hz", f"{f}",
                ]
                jobs.append(dict(
                    block="fa", fmu="fast", t_end=400.0,
                    tag=f"fa_f{_tag(f, '.3f')}_{sup}",
                    args=BASE + ["--fix-leogo-xqt"] + SUPPORT[sup] + dist,
                ))

    if "torsion" in blocks:
        for sup in ("off", "on"):
            for f in TORSION_FREQS:
                dist = [
                    "--load-step-mw", "3.0", "--event-time", "10",
                    "--event-duration", "40", "--load-ramp-on-s", "1",
                    "--load-sine-mean", "0.0", "--load-sine-amplitude", "1.0",
                    "--load-sine-freq-hz", f"{f}",
                ]
                jobs.append(dict(
                    block="torsion", fmu="fast", t_end=40.0,
                    tag=f"tors_f{_tag(f, '.2f')}_{sup}",
                    args=BASE + ["--fix-leogo-xqt"] + SUPPORT[sup] + dist,
                ))

    if "stability" in blocks:
        # Keep the LEOGO ~5.3 Hz q-axis artifact IN (no --fix-leogo-xqt) so the
        # droop can interact with it. Perturb with a load-step kick.
        for droop in STAB_DROOP:
            for lpf in STAB_LPF:
                dist = [
                    "--load-step-mw", "10.0", "--event-time", "10",
                    "--event-duration", "45", "--load-ramp-on-s", "0.5",
                ]
                support = [
                    "--droop-nm-per-hz", f"{droop:.0f}",
                    "--inertia-nm-s-per-hz", "0",
                    "--freq-lpf-hz", f"{lpf}", "--freq-lpf-order", "2",
                    "--support-start", "10",
                ]
                jobs.append(dict(
                    block="stability", fmu="fast", t_end=60.0,
                    tag=f"stab_d{_tag(droop, '.0e')}_lpf{_tag(lpf, '.2f')}",
                    args=BASE + support + dist,
                ))

    for j in jobs:
        j["out"] = OUT_ROOT / j["block"] / f"{j['tag']}.csv"
    return jobs


def run_job(job: dict, *, log_dir: Path) -> tuple[bool, float]:
    """Run one sweep point as a subprocess. Returns (ok, wall_seconds)."""
    out_rel = job["out"].relative_to(PROJECT_ROOT)
    cmd = [sys.executable, str(DRIVER),
           "--t-end", f"{job['t_end']}", "--fmu", job["fmu"],
           "--out", str(out_rel)] + job["args"]

    job["out"].parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['tag']}.log"

    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=log,
                              stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    ok = proc.returncode == 0 and job["out"].exists()
    return ok, wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", choices=["ss", "fa", "torsion", "stability", "all"],
                    default="all")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the jobs and time estimate; run nothing.")
    ap.add_argument("--smoke", action="store_true",
                    help="Run one short point per FMU variant to validate the "
                         "pipeline (writes into full_matrix/_smoke).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run points whose CSV already exists.")
    args = ap.parse_args()

    blocks = {"ss", "torsion", "stability"} if args.block == "all" else {args.block}

    if args.smoke:
        _run_smoke()
        return

    jobs = build_jobs(blocks)
    pending = [j for j in jobs if args.force or not j["out"].exists()]
    done = len(jobs) - len(pending)
    est_min = sum(j["t_end"] * WALL_PER_SIMS for j in pending) / 60.0

    print(f"Full-matrix sweep: {len(jobs)} points "
          f"({done} already done, {len(pending)} to run)")
    print(f"Estimated remaining wall time: {est_min:.0f} min "
          f"(~{est_min/60:.1f} h) at {WALL_PER_SIMS:.1f} s/sim-s\n")
    for j in jobs:
        mark = "skip" if (j not in pending) else "RUN "
        print(f"  [{mark}] {j['block']:>9} {j['tag']:<26} "
              f"t_end={j['t_end']:.0f}s fmu={j['fmu']}")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    log_dir = OUT_ROOT / "_logs"
    n_ok = n_fail = 0
    for i, j in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {j['block']} {j['tag']} "
              f"(t_end={j['t_end']:.0f}s) ...", flush=True)
        try:
            ok, wall = run_job(j, log_dir=log_dir)
        except Exception as exc:                      # keep the sweep alive
            print(f"    ERROR launching: {exc}", flush=True)
            n_fail += 1
            continue
        if ok:
            n_ok += 1
            print(f"    ok  ({wall:.0f}s)  -> {j['out'].name}", flush=True)
        else:
            n_fail += 1
            print(f"    FAILED (see {log_dir / (j['tag'] + '.log')})", flush=True)

    print(f"\nDone. {n_ok} ok, {n_fail} failed. "
          f"Re-run to resume any that failed.")


def _run_smoke() -> None:
    """One short torsion (fast.fmu) and one short SS (fast_debug.fmu) point."""
    smoke_dir = OUT_ROOT / "_smoke"
    probes = [
        dict(block="torsion", fmu="fast", t_end=20.0, tag="smoke_tors",
             args=BASE + ["--fix-leogo-xqt"] + SUPPORT["on"] + [
                 "--load-step-mw", "3.0", "--event-time", "5",
                 "--event-duration", "20", "--load-sine-mean", "0.0",
                 "--load-sine-amplitude", "1.0", "--load-sine-freq-hz", "3.05"]),
        dict(block="ss", fmu="debug", t_end=20.0, tag="smoke_ss",
             args=BASE + ["--fix-leogo-xqt"] + SUPPORT["off"] + [
                 "--load-step-mw", "3.0", "--event-time", "5",
                 "--event-duration", "20", "--load-sine-mean", "0.0",
                 "--load-sine-amplitude", "1.0", "--load-sine-freq-hz", "0.229"]),
    ]
    for p in probes:
        p["out"] = smoke_dir / f"{p['tag']}.csv"
    print("Smoke test: 2 short runs to validate the pipeline "
          f"(~{sum(p['t_end'] for p in probes) * WALL_PER_SIMS / 60:.1f} min)\n")
    for p in probes:
        print(f"  {p['tag']} (fmu={p['fmu']}, t_end={p['t_end']:.0f}s) ...",
              flush=True)
        ok, wall = run_job(p, log_dir=smoke_dir / "_logs")
        status = "ok" if ok else f"FAILED (see {smoke_dir/'_logs'/(p['tag']+'.log')})"
        print(f"    {status}  ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
