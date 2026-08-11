"""Resonance curve of the tower SS mode via the LEOGO grid load path.

Sweeps the frequency of a sustained sinusoidal active-power load modulation at
the LEOGO main bus and measures the settled tower side-to-side (SS) and fore-aft
(FA) response amplitude at each frequency. Because the SS mode is driven through
the genuine grid -> Pe -> Te -> SS coupling, a sharp peak at ~0.234 Hz
demonstrates the frequency-selective electro-mechanical resonance, while the
thrust-driven FA mode stays flat.

This reuses the reduced LEOGO + WindTurbineTower model (no OpenFAST FMU) from
test_WT_LEOGO_tower_sim.py, running each probe frequency in-process.

Usage
-----
    python casestudies/dyn_sim/sweep_ss_resonance.py                 # run + plot
    python casestudies/dyn_sim/sweep_ss_resonance.py --plot-only     # re-plot
"""
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops_openfast.dyn_models as ext_lib

from test_WT_LEOGO_tower_sim import build_model, scalar


def lockin(t, sig, f, t_lo, t_hi):
    """Amplitude of sig at frequency f over [t_lo, t_hi] (quadrature demod)."""
    m = (t >= t_lo) & (t <= t_hi)
    tt = t[m]
    s = sig[m].astype(float)
    s = s - s.mean()
    w = 2.0 * np.pi * f
    c = np.trapezoid(s * np.cos(w * tt), tt)
    q = np.trapezoid(s * np.sin(w * tt), tt)
    span = tt[-1] - tt[0]
    return 2.0 * np.hypot(c, q) / span


def run_probe(f_mod, grid_mod_amp, mod_start, t_end, dt):
    """Run one single-frequency grid-load probe; return (t, ss, fa, gfreq)."""
    model = build_model(ss_enable=1, fa_enable=1, droop_enable=0)
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)

    wt_model = ps.windturbine_tower['WindTurbineTower']
    uic_model = ps.vsc['UIC_sig']
    gen_model = ps.gen['GEN']

    wind_speed_initial = wt_model.wind_speed_init()
    p_ref_initial = wt_model.P_ref_from_wind(wind_speed_initial, uic_model.par['S_n'])
    uic_model.par['p_ref'][:] = p_ref_initial
    uic_model.par['q_ref'][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()

    s_base_mva = float(model['base_mva'])
    load_bus_idx = gen_model.bus_idx_red['terminal'][0]
    y_grid_mod = grid_mod_amp / s_base_mva
    w_mod = 2.0 * np.pi * f_mod

    def grid_mod_scale(t):
        if t < mod_start:
            return 0.0
        return float(np.sin(w_mod * (t - mod_start)))

    def set_load_step(t):
        ps.y_bus_red_mod[(load_bus_idx, load_bus_idx)] = grid_mod_scale(t) * y_grid_mod

    f_n_grid = float(np.asarray(gen_model.sys_par['f_n']).ravel()[0])

    def f_ode(t, x):
        wt_model._sim_time = t
        set_load_step(t)
        gen_speed_pu = np.asarray(gen_model.speed(x, None), dtype=float).ravel()
        wt_model.set_grid_frequency_hz(f_n_grid * (1.0 + float(np.mean(gen_speed_pu))))
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, t_end, max_step=dt)

    f_n = f_n_grid
    ts, ss, fa, gf = [], [], [], []

    def log(t, x, v):
        ts.append(t)
        ss.append(scalar(wt_model.ss_acceleration(x, v)))
        fa.append(scalar(wt_model.fa_acceleration(x, v)))
        gspd = np.asarray(gen_model.speed(x, v), dtype=float).ravel()
        gf.append(f_n * (1.0 + float(np.mean(gspd))))

    wt_model._sim_time = 0.0
    set_load_step(0.0)
    v0 = ps.solve_algebraic(0.0, x0)
    log(0.0, x0, v0)

    while solver.t < t_end:
        solver.step()
        t = solver.t
        x = solver.x
        wt_model._sim_time = t
        set_load_step(t)
        v = ps.solve_algebraic(t, x)
        log(t, x, v)

    return (np.asarray(ts), np.asarray(ss), np.asarray(fa), np.asarray(gf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-mod-amp", type=float, default=5.0,
                    help="Load modulation amplitude [MW].")
    ap.add_argument("--mod-start", type=float, default=5.0)
    ap.add_argument("--t-end", type=float, default=110.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--settle", type=float, default=60.0,
                    help="Lock-in window start [s] (after build-up).")
    ap.add_argument("--freqs", type=float, nargs="+", default=[
        0.200, 0.210, 0.216, 0.222, 0.226, 0.230, 0.234,
        0.238, 0.242, 0.248, 0.255, 0.265, 0.280, 0.300,
    ])
    ap.add_argument("--csv", default="results/tower_test/ss_resonance_curve.csv")
    ap.add_argument("--out", default="results/tower_test/ss_resonance_curve.png")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        recs = []
        for i, f in enumerate(args.freqs):
            t0 = time.perf_counter()
            t, ss, fa, gf = run_probe(f, args.grid_mod_amp, args.mod_start,
                                      args.t_end, args.dt)
            ss_amp = lockin(t, ss, f, args.settle, t[-1])
            fa_amp = lockin(t, fa, f, args.settle, t[-1])
            gf_amp = lockin(t, gf, f, args.settle, t[-1])
            recs.append(dict(freq_hz=f, ss_amp=ss_amp, fa_amp=fa_amp,
                             gfreq_amp=gf_amp))
            print(f"[{i+1:2d}/{len(args.freqs)}] f={f:.3f} Hz  "
                  f"SS={ss_amp:.4f}  FA={fa_amp:.4f}  "
                  f"gf={1000*gf_amp:.2f} mHz  ({time.perf_counter()-t0:.1f}s)")
        pd.DataFrame(recs).to_csv(csv_path, index=False)
        print(f"Sweep written to: {csv_path}")

    d = pd.read_csv(csv_path)
    f = d["freq_hz"].to_numpy()
    ss = d["ss_amp"].to_numpy()
    fa = d["fa_amp"].to_numpy()

    i_pk = int(np.argmax(ss))
    print("=" * 56)
    print(f"SS resonance peak: {ss[i_pk]:.4f} m/s^2 at {f[i_pk]:.3f} Hz")
    print(f"SS/FA at peak    : {ss[i_pk]/max(fa[i_pk],1e-12):.1f} x")
    print("=" * 56)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(f, ss, "o-", color="tab:red", label="SS (side-to-side)")
    ax.plot(f, fa, "s-", color="tab:green", label="FA (fore-aft)")
    ax.axvline(0.234, color="k", ls="--", lw=0.9, alpha=0.6,
               label="SS natural freq 0.234 Hz")
    ax.set_xlabel("Grid load modulation frequency [Hz]")
    ax.set_ylabel("Settled tower accel amplitude [m/s$^2$]")
    ax.set_title(f"Tower resonance curve: SS response to a "
                 f"$\\pm${args.grid_mod_amp:.0f} MW cyclic grid load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Figure written to: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
