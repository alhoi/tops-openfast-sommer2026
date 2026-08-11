r"""
Isolated modal identification of the OpenFAST (FMU) wind turbine.

The FMU turbine + UIC are connected to a stiff INFINITE BUS instead of the
LEOGO grid (model: casestudies/ps_data/test_WT_FMU_drivetrain_.py, generator
'IB' with H = 1e5). Because the OpenFAST turbine is a compiled FMU, its
structural modes (tower side-to-side / fore-aft, drivetrain torsion) are NOT
states of the power-system model and cannot be obtained by linearisation. They
are instead identified from the FMU's own time-domain FREE RING-DOWN: the
turbine is initialised away from structural equilibrium, so each enabled mode
rings at its natural frequency and decays with its own damping. We record the
FMU structural outputs, take an FFT for the natural frequency, and fit the decay
envelope (log-decrement) for the damping ratio.

Which modes appear depends on the ElastoDyn DOF flags in
test1002/IEA-15-240-RWT-Monopile_ElastoDyn.dat:
  TwSSDOF1 -> tower side-to-side (YawBrTAyp),
  TwFADOF1 -> tower fore-aft     (YawBrTAxp),
  DrTrDOF  -> drivetrain torsion (HSShftTq).
Enable the DOFs you want to identify before running; a flat signal below means
the corresponding DOF is disabled.

Run (takes a few minutes; ZMQ_Mode=1 ROSCO needs the zero-support responder):
    .\.venv\Scripts\python.exe casestudies\modal_analysis\modal_openfast_isolated.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "casestudies" / "dyn_sim"))

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops_openfast.dyn_models as ext_lib
import casestudies.ps_data.test_WT_FMU_drivetrain_ as model_data
from rosco_zmq_grid_coupling import GridTorqueResponder

T_END = 120.0          # s of free ring-down to record
SETTLE = 3.0           # s skipped at the start (FMU priming glitch)

# A short generator-torque impulse (via the ZMQ channel) rings the drivetrain
# cleanly for the torsion estimate; the init transient already excites the tower.
PULSE_T0 = 40.0        # s
PULSE_DUR = 0.20       # s
PULSE_NM = 3.0e6       # Nm

# (label, FMU output, spectrum band [Hz], analysis window [s])
SIGNALS = [
    ("Tower side-to-side", "YawBrTAyp", (0.10, 0.60), (SETTLE, PULSE_T0 - 2.0)),
    ("Tower fore-aft",     "YawBrTAxp", (0.10, 0.60), (SETTLE, PULSE_T0 - 2.0)),
    ("Drivetrain torsion", "HSShftTq",  (2.00, 5.00), (PULSE_T0 + 0.15, PULSE_T0 + 8.0)),
]

# Physical units of each FMU output (OpenFAST defaults): tower-top accelerations
# in m/s^2, high-speed-shaft torque in kN.m.
UNITS = {
    "YawBrTAyp": r"m/s$^2$",
    "YawBrTAxp": r"m/s$^2$",
    "HSShftTq":  r"kN$\cdot$m",
}


def spectrum(t, y):
    dt = float(np.median(np.diff(t)))
    y = (y - y.mean()) * np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y)) / y.size * 2.0
    freq = np.fft.rfftfreq(y.size, dt)
    return freq, spec


def detrend(t, y, deg=3):
    """Remove a low-order polynomial trend (the slow settling transient)."""
    if t.size < deg + 2:
        return y - y.mean()
    return y - np.polyval(np.polyfit(t - t[0], y, deg), t - t[0])


def peak_freq(t, y, band):
    f, s = spectrum(t, y)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or s[m].max() <= 0:
        return float("nan"), (f, s)
    return float(f[m][np.argmax(s[m])]), (f, s)


def damping_ratio(t, y, f0):
    """Log-decrement on the successive peaks of a decaying sinusoid."""
    if not np.isfinite(f0) or f0 <= 0:
        return float("nan")
    yy = y - y.mean()
    env = np.abs(yy)
    # local maxima of the rectified signal
    idx = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]))[0] + 1
    if idx.size < 4:
        return float("nan")
    tp, ap = t[idx], env[idx]
    keep = ap > 0.15 * ap.max()          # ignore noise-level peaks
    tp, ap = tp[keep], ap[keep]
    if tp.size < 4:
        return float("nan")
    slope = np.polyfit(tp - tp[0], np.log(ap), 1)[0]   # ln A = ln A0 + slope*t
    zeta = -slope / (2.0 * np.pi * f0)
    return float(zeta)


def analyze(t, rec) -> None:
    out_png = PROJECT_ROOT / "results" / "modal" / "openfast_isolated_ringdown.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SIGNALS), 2, figsize=(12.0, 9.0))

    print("\n=== Isolated OpenFAST FMU - identified structural modes ===")
    print(f"{'mode':>20}  {'f [Hz]':>9}  {'zeta [%]':>9}  {'amp':>10}")
    for row, (label, name, band, win) in enumerate(SIGNALS):
        m = (t >= win[0]) & (t <= win[1])
        tt = t[m]
        y = detrend(tt, np.asarray(rec[name])[m])   # remove settling transient
        amp = 0.5 * float(np.ptp(y)) if y.size else float("nan")
        f0, (f, s) = peak_freq(tt, y, band)
        zeta = damping_ratio(tt, y, f0)
        flat = amp < 1e-6
        print(f"{label:>20}  {f0:9.4f}  {100*zeta:9.3f}  {amp:10.3e}"
              + ("   (DOF off?)" if flat else ""))

        unit = UNITS.get(name, "")
        ax_t, ax_f = axes[row]
        ax_t.plot(tt, y, lw=0.8, color="#1f77b4")
        ax_t.set_ylabel(f"{label} [{unit}]\n(detrended)")
        ax_t.grid(True, alpha=0.3)
        ax_f.plot(f, s, lw=1.0, color="#0b6e4f")
        ax_f.set_ylabel(f"Amplitude [{unit}]")
        if np.isfinite(f0):
            ax_f.axvline(f0, color="#d62728", ls="--", lw=1.0)
            ax_f.set_title(f"{f0:.3f} Hz,  zeta = {100*zeta:.2f}%", fontsize=10)
        ax_f.set_xlim(0, band[1] * 1.5)
        ax_f.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle("OpenFAST FMU (isolated, infinite bus) - free ring-down modal ID",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved ring-down figure: {out_png}")


def main() -> None:
    model = model_data.load()

    # test_WT_FMU_drivetrain_ uses fast.fmu, which exposes only the fore-aft
    # tower accel (YawBrTAxp). Point it at fast_debug.fmu, which additionally
    # exposes the side-to-side accel (YawBrTAyp), so all three structural modes
    # can be identified from one run.
    _row = model["FMUtoUICdrivetrain"]["FMUtoUICdrivetrain"]
    _hdr, _val = _row[0], _row[1]
    _debug = PROJECT_ROOT / "fast_debug.fmu"
    if _debug.exists():
        _val[_hdr.index("FMU_path")] = str(_debug)
        _val[_hdr.index("fmu_filename")] = "fast_debug.fmu"

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)
    fmu_model = ps.FMUtoUICdrivetrain["FMUtoUICdrivetrain"]
    uic = ps.vsc["UIC_sig"]

    # Zero-support responder so the ZMQ_Mode=1 ROSCO gets an immediate 0 reply
    # instead of blocking (support disabled: gains 0, start beyond t_end).
    responder = GridTorqueResponder(
        droop_nm_per_hz=0.0, inertia_nm_s_per_hz=0.0,
        support_start_s=T_END + 100.0,
        log_path=str(PROJECT_ROOT / "results" / "modal" / "_zmq_idle_log.csv"),
    )
    responder.start()

    uic.par["p_ref"][:] = 0.60      # rough PF guess; FMU sets the true P_ref
    uic.par["q_ref"][:] = 0.0
    ps.power_flow()
    ps.init_dyn_sim()               # primes the FMU

    x0 = ps.x0.copy()
    dt = float(fmu_model._fmu_dt)

    def f_ode(t_, x_):
        return ps.state_derivatives(t_, x_, ps.solve_algebraic(t_, x_))

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, T_END, max_step=dt)

    rec = {"t": []}
    for _, name, _, _ in SIGNALS:
        rec[name] = []

    print(f"Running isolated FMU ring-down for {T_END:.0f} s "
          f"(torque impulse at t={PULSE_T0:.0f}s) ...", flush=True)
    while solver.t < T_END:
        solver.step()
        x, t = solver.x, solver.t
        v = ps.solve_algebraic(t, x)
        # Open-loop generator-torque impulse to ring the drivetrain torsion.
        pulse = PULSE_NM if (PULSE_T0 <= t < PULSE_T0 + PULSE_DUR) else 0.0
        responder.set_manual_offset(pulse)
        fmu_model.step_fmu(x, v, t, dt)
        out = fmu_model.get_all_fmu_outputs()
        rec["t"].append(t)
        for _, name, _, _ in SIGNALS:
            rec[name].append(float(out.get(name, np.nan)))
        if int(t) != int(t - dt):
            print(f"\r  t = {t:5.1f} / {T_END:.0f} s", end="", flush=True)
    print()

    responder.close()

    t = np.asarray(rec["t"])

    # Save the raw time series for offline re-analysis (no re-run needed).
    csv_out = PROJECT_ROOT / "results" / "modal" / "openfast_isolated_ringdown.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rec).to_csv(csv_out, index=False)
    print(f"Saved raw time series: {csv_out}")

    analyze(t, rec)


if __name__ == "__main__":
    if "--from-csv" in sys.argv:
        _csv = PROJECT_ROOT / "results" / "modal" / "openfast_isolated_ringdown.csv"
        _df = pd.read_csv(_csv)
        _rec = {c: _df[c].to_numpy() for c in _df.columns}
        analyze(np.asarray(_rec["t"]), _rec)
    else:
        main()
