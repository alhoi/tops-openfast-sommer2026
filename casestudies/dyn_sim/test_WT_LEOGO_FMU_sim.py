"""
LEOGO + high-fidelity OpenFAST wind turbine (FMU co-simulation).

This is the FMU counterpart of ``test_WT_LEOGO_sim.py``.

``test_WT_LEOGO_sim.py`` replaces the original LEOGO WTG1 converter with the
*simplified* analytic WindTurbine + UIC_sig. That turbine is a normal DAE
model: its drivetrain / pitch / aero states are integrated by the TOPS
solver together with the rest of the electrical network.

Here we instead drop in the *high-fidelity* OpenFAST turbine
(``FMUtoUICdrivetrain`` + UIC_sig). The electrical interface to the grid is
identical - the FMU turbine connects to the UIC through the same three
connections (P_e, S_n_UIC in; P_ref out). The difference is that the
aero-servo-elastic dynamics live inside the OpenFAST FMU, which is a
co-simulation slave and must be stepped by hand once per network step.

Two consequences for the simulation loop (compared with the simplified sim):

  1. Fixed time step. The OpenFAST FMU cannot take variable communication
     steps, so dt is locked to fmu_dt (0.01 s) and every RK4 macro-step
     advances the FMU exactly once.
  2. Explicit co-simulation ordering. After each network step we call
     mdl.step_fmu(x, v, t, dt), and at the end mdl.terminate_fmu().
     We also chdir to the project root so the FMU's relative paths
     (openfast_fmu/ extraction dir and wd.txt) resolve.
"""

from pathlib import Path
from copy import deepcopy
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# .../tops-openfast-sommer2026
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The OpenFAST FMU extracts to openfast_fmu/ and reads wd.txt using paths
# relative to the current working directory, so run from the project root.
os.chdir(PROJECT_ROOT)

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops_openfast.dyn_models as ext_lib

from LEOGO.LEOGO_ps import load as load_leogo


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def _resolve_fmu_path() -> str:
    """Return the first FMU file that exists.

    Prefer fast_debug.fmu: it exposes the tower-top side-to-side acceleration
    output (YawBrTAyp) in addition to fore-aft (YawBrTAxp). The regular
    fast.fmu only exposes YawBrTAxp, so it cannot report the side-to-side
    tower mode (TwSSDOF1).
    """
    candidates = [
        PROJECT_ROOT / "OpenFAST" / "fast_debug.fmu",
        PROJECT_ROOT / "fast_debug.fmu",
        PROJECT_ROOT / "OpenFAST" / "fast.fmu",
        PROJECT_ROOT / "fast.fmu",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    # Fall back to the repo-root name so the error message is meaningful.
    return str(candidates[-1])


def build_model():
    """
    Start from the LEOGO electrical network and replace its original,
    simplified WT1 grid-side converter with the OpenFAST FMU turbine
    (FMUtoUICdrivetrain) driving the UIC_sig converter.

    This mirrors build_model() in test_WT_LEOGO_sim.py; only the turbine
    block differs (FMUtoUICdrivetrain instead of windturbine).
    """
    model = deepcopy(load_leogo())

    wt_bus = "Busbar WTG1 LV"

    # -------------------------------------------------------------
    # 1. Remove the original LEOGO converter at WTG1.
    #    Its former component name was WT1_LEOGO; we reuse that name
    #    for the UIC_sig so the FMU turbine can connect to it.
    # -------------------------------------------------------------
    old_converter = model["vsc"].pop("GridSideConverter_PV", None)
    if old_converter is None:
        raise KeyError(
            "Could not find model['vsc']['GridSideConverter_PV']. "
            "Check LEOGO_ps.py before running."
        )

    # -------------------------------------------------------------
    # 2. Add the UIC model at the same physical electrical bus.
    #    Identical to the simplified LEOGO sim.
    # -------------------------------------------------------------
    model["vsc"]["UIC_sig"] = [
        [
            "name", "bus", "S_n", "V_n",
            "v_ref", "p_ref", "q_ref",
            "Ki", "Kv", "xf",
            "perfect_tracking", "T_filter",
        ],
        [
            "WT1_LEOGO",   # UIC component name
            wt_bus,        # LEOGO electrical connection point
            20.0,          # MVA
            0.69,          # kV: Busbar WTG1 LV is a 690 V bus
            1.05,
            0.0,           # Set from the FMU operating point before power flow
            0.0,
            0.03,
            0.0,
            0.1,
            0,             # perfect_tracking (0 = grid disturbance reaches the WT)
            0.01,
        ],
    ]

    # -------------------------------------------------------------
    # 3. Add the OpenFAST FMU turbine in place of the analytic
    #    windturbine block. It connects to the UIC named WT1_LEOGO.
    #
    #    The parameter row matches casestudies/ps_data/
    #    test_WT_FMU_drivetrain_.py, except:
    #      * V_n is 0.69 kV (the LEOGO WTG1 LV bus), not 22 kV.
    #      * openfast_test_dir / wd_path point at this repo so the FMU
    #        finds the test1002/ OpenFAST case (testNr = 1002).
    # -------------------------------------------------------------
    fmu_path = _resolve_fmu_path()

    model["FMUtoUICdrivetrain"] = {
        "FMUtoUICdrivetrain": [
            [
                "name", "UIC", "S_n", "V_n",
                "FMU_path", "fmu_filename", "control_mode",
                "wd_path", "openfast_test_dir", "testNr",
                "J_m", "J_e", "K", "D",
                "omega_m_rated", "fmu_dt", "ElecPwrCom_kW", "efficiency",
                "speed_lpf_type", "speed_lpf_corner_rad_s", "speed_lpf_damping",
            ],
            [
                "FMUtoUICdrivetrain1",
                "WT1_LEOGO",       # Must match UIC_sig name above
                15.0,              # MVA (WT local base)
                0.69,              # kV at Busbar WTG1 LV
                fmu_path,
                "fast.fmu",
                3,                 # control_mode (torque coupling)
                str(PROJECT_ROOT / "openfast_fmu" / "resources" / "wd.txt"),
                str(PROJECT_ROOT), # directory that CONTAINS test1002/
                1002,              # testNr -> selects test1002 case
                352460500.0,       # J_m
                1836784.0,         # J_e
                69737644900.0 / 100.0,  # K
                35698200.0 / 10.0,      # D
                7.559987120819503,      # omega_m_rated (rpm)
                0.01,                   # fmu_dt (s) -> also the network dt
                6650.0,                 # ElecPwrCom_kW (MPPT command; ~6.6 MW at 8 m/s, Region 2)
                0.95756,                # efficiency
                2,
                1.00810,
                0.70000,
            ],
        ],
    }

    return model


# ---------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------

def scalar(value):
    """Return a scalar float from a scalar or length-one NumPy array."""
    return float(np.asarray(value).reshape(-1)[0])


def grid_frequency_hz(gen_model, x, v):
    """Inertia-weighted centre-of-inertia grid frequency [Hz].

    Mirrors the f_grid_hz computation in collect_results, factored out so the
    in-process ZMQ torque coupling can read the live grid frequency each step.
    """
    gen_speed_dev = np.asarray(gen_model.speed(x, v), dtype=float)
    gen_s_n = np.asarray(gen_model.par["S_n"], dtype=float)
    gen_H = np.asarray(gen_model.par["H"], dtype=float)
    inertia_w = gen_H * gen_s_n
    f_n = float(np.atleast_1d(gen_model.sys_par["f_n"])[0])
    omega_grid_pu = float(np.sum((1.0 + gen_speed_dev) * inertia_w) / np.sum(inertia_w))
    return omega_grid_pu * f_n


def collect_results(ps, fmu_model, uic_model, gen_model, t, x, v):
    """
    Store key grid-side signals plus the FMU turbine outputs.

    Unlike the analytic turbine, the aero / rotor / pitch quantities are not
    DAE states here: they are read back from the OpenFAST FMU outputs.
    """
    sys_s_n = float(ps.sys_data["s_n"])
    uic_s_n = float(uic_model.par["S_n"][0])

    # UIC terminal variables (grid interface)
    v_terminal = uic_model.v_t(x, v)[0]
    s_uic = uic_model.s_e(x, v)[0]
    i_a = uic_model.i_a(x, v)[0]

    uic_states = uic_model.local_view(x)
    vi = uic_states["vi_x"][0] + 1j * uic_states["vi_y"][0]

    # Total synchronous-generator electrical output (LEOGO gas turbines)
    p_gen_local = np.asarray(gen_model.p_e(x, v), dtype=float)
    q_gen_local = np.asarray(gen_model.q_e(x, v), dtype=float)
    gen_s_n = np.asarray(gen_model.par["S_n"], dtype=float)
    p_gen_total_sys_pu = float(np.sum(p_gen_local * gen_s_n / sys_s_n))
    q_gen_total_sys_pu = float(np.sum(q_gen_local * gen_s_n / sys_s_n))

    # Grid (system) frequency from the synchronous generators. The generator
    # "speed" state is the per-unit deviation from synchronous speed, so the
    # electrical frequency is (1 + speed) * f_n. Use the inertia-weighted
    # centre-of-inertia (COI) speed as the single system-frequency signal.
    gen_speed_dev = np.asarray(gen_model.speed(x, v), dtype=float)
    gen_H = np.asarray(gen_model.par["H"], dtype=float)
    inertia_w = gen_H * gen_s_n
    f_n = float(np.atleast_1d(gen_model.sys_par["f_n"])[0])
    omega_grid_pu = float(np.sum((1.0 + gen_speed_dev) * inertia_w) / np.sum(inertia_w))
    f_grid_hz = omega_grid_pu * f_n

    # FMU turbine coupling state (electrical-side drivetrain speed)
    fmu_states = fmu_model.local_view(x)
    omega_e_pu = scalar(fmu_states["omega_e"])

    row = {
        "t": t,

        # WT electrical output at the UIC / grid
        "P_e_sys_pu": scalar(fmu_model.P_e(x, v)) * uic_s_n / sys_s_n,
        "P_ref_sys_pu": scalar(fmu_model.P_ref(x, v)) * uic_s_n / sys_s_n,
        "P_uic_bus_sys_pu": s_uic.real * uic_s_n / sys_s_n,
        "Q_uic_bus_sys_pu": s_uic.imag * uic_s_n / sys_s_n,
        "omega_e_pu": omega_e_pu,

        # Grid (system) frequency from the LEOGO synchronous generators
        "omega_grid_pu": omega_grid_pu,
        "f_grid_hz": f_grid_hz,

        # UIC / bus at Busbar WTG1 LV
        "V_WTG1_LV_pu": abs(v_terminal),
        "angle_WTG1_LV_deg": np.degrees(np.angle(v_terminal)),
        "V_uic_internal_pu": abs(vi),
        "I_uic_pu": abs(i_a),
        "I_uic_angle_deg": np.degrees(np.angle(i_a)),

        # Remaining LEOGO synchronous generation
        "P_sync_generators_total_sys_pu": p_gen_total_sys_pu,
        "Q_sync_generators_total_sys_pu": q_gen_total_sys_pu,
    }

    # High-fidelity OpenFAST outputs (rotor speed, pitch, wind, torque, ...).
    # YawBrTAxp / YawBrTAyp are the tower-top (yaw-bearing) fore-aft and
    # side-to-side translational accelerations - the signals used to identify
    # the fore-aft (TwFADOF1) and side-to-side (TwSSDOF1) tower modes.
    # Note: YawBrTAyp is only available from fast_debug.fmu.
    if hasattr(fmu_model, "get_all_fmu_outputs"):
        fmu_out = fmu_model.get_all_fmu_outputs()
        for key in ("RotSpeed", "GenSpeed", "GenTq", "BldPitch1",
                    "Wind1VelX", "YawBrTAxp", "YawBrTAyp", "HSShftTq"):
            if key in fmu_out:
                row[f"fmu_{key}"] = float(fmu_out[key])

    return row


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LEOGO network with the OpenFAST FMU wind turbine at WTG1."
    )
    parser.add_argument("--t-end", type=float, default=60.0)
    # dt is forced to fmu_dt below; exposed only for completeness.
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (absolute, or relative to the project root). "
             "Defaults to results/WT1_LEOGO_FMU_results.csv.",
    )
    parser.add_argument(
        "--fmu",
        choices=["auto", "fast", "debug"],
        default="auto",
        help="Which OpenFAST FMU to use. 'auto' keeps the default resolution "
             "(prefers fast_debug.fmu, needed for the side-to-side tower output "
             "YawBrTAyp). 'fast' forces fast.fmu (has HSShftTq for the drivetrain "
             "torsion mode, faster, no per-step ROSCO debug writes). 'debug' "
             "forces fast_debug.fmu.",
    )

    # Smooth load step applied as a temporary shunt at the main grid bus.
    parser.add_argument("--load-step-mw", type=float, default=5.0)
    parser.add_argument("--load-step-mvar", type=float, default=0.0)
    parser.add_argument("--event-time", type=float, default=10.0)
    parser.add_argument("--event-duration", type=float, default=30.0)
    parser.add_argument(
        "--load-ramp-on-s",
        type=float,
        default=2.0,
        help="Duration of the smooth load increase. Zero gives an ideal step.",
    )
    parser.add_argument(
        "--load-ramp-off-s",
        type=float,
        default=2.0,
        help="Duration of the smooth load removal. Zero gives an ideal step.",
    )

    # Optional sinusoidal modulation of the load, on top of the step envelope.
    # Applied load(t) = load_step_mw * envelope(t)
    #                   * (load_sine_mean + load_sine_amplitude * sin(2*pi*f*(t - t_on) + phase))
    # Default amplitude 0 keeps the plain smooth step (backward compatible).
    # Set --load-sine-freq-hz to the tower fore-aft mode (~0.236 Hz) to probe
    # for resonant electromechanical interaction with the OpenFAST tower DOF.
    parser.add_argument("--load-sine-freq-hz", type=float, default=0.236)
    parser.add_argument("--load-sine-amplitude", type=float, default=0.0)
    parser.add_argument("--load-sine-mean", type=float, default=1.0)
    parser.add_argument("--load-sine-phase-deg", type=float, default=0.0)

    # Optional diagnostic modulation of the coupling (generator) torque command.
    # Unlike the grid load (which reaches the drivetrain only weakly, via the
    # UIC and the electrical power), this directly modulates the electromagnetic
    # torque applied to the OpenFAST rotor at a chosen frequency. Driving at the
    # side-to-side tower mode (~0.236 Hz) is the most efficient way to force
    # that mechanical mode. Default amplitude 0 disables it (torque unchanged).
    parser.add_argument(
        "--torque-mod-amplitude",
        type=float,
        default=0.0,
        help="Fractional amplitude of the sinusoidal torque-command modulation "
             "(e.g. 0.1 = +/-10%% of the coupling torque).",
    )
    parser.add_argument("--torque-mod-freq-hz", type=float, default=0.236)
    parser.add_argument("--torque-mod-start", type=float, default=10.0)

    # Optional diagnostic modulation of the demanded electrical power
    # (ElecPwrCom) command. With ROSCO active (VSContrl=5) the external torque
    # command is ignored, but ElecPwrCom is the grid-side power demand the
    # controller does respond to. Modulating it at the side-to-side tower
    # frequency represents a grid/converter power-command oscillation and is
    # the physically effective way to make the controller vary the generator
    # torque, exciting the side-to-side tower mode. Default amplitude 0 = off.
    parser.add_argument(
        "--elecpwr-mod-amplitude",
        type=float,
        default=0.0,
        help="Fractional amplitude of the sinusoidal ElecPwrCom modulation "
             "(e.g. 0.1 = +/-10%% of the demanded electrical power).",
    )
    parser.add_argument("--elecpwr-mod-freq-hz", type=float, default=0.236)
    parser.add_argument("--elecpwr-mod-start", type=float, default=10.0)

    # Optional three-phase fault at the WT connection point
    parser.add_argument("--fault", action="store_true")
    parser.add_argument("--fault-start", type=float, default=10.0)
    parser.add_argument("--fault-duration", type=float, default=0.05)
    parser.add_argument("--fault-admittance", type=float, default=1e6)

    # Live grid-driven generator-torque coupling through ROSCO's ZeroMQ
    # interface. When enabled, an in-process ZeroMQ server answers ROSCO each
    # step with a generator-torque offset derived from the LIVE LEOGO grid
    # frequency (frequency-support droop + synthetic inertia), closing a genuine
    # electrical -> mechanical loop. Requires a ZMQ-enabled ROSCO libDISCON.dll
    # (patched to apply ZMQ_TorqueOffset) and ROSCO.IEA15MW.IN with ZMQ_Mode=1,
    # ZMQ_UpdatePeriod = fmu_dt. Set the droop/inertia gains to zero for the
    # coupling-OFF reference run.
    parser.add_argument("--zmq-grid", action="store_true",
                        help="Enable in-process grid-driven ROSCO torque coupling.")
    parser.add_argument("--droop-nm-per-hz", type=float, default=2.0e7,
                        help="Frequency-support droop gain [Nm per Hz].")
    parser.add_argument("--inertia-nm-s-per-hz", type=float, default=0.0,
                        help="Synthetic-inertia gain [Nm.s per Hz].")
    parser.add_argument("--deload-nm", type=float, default=0.0,
                        help="Standing de-load (curtailment) torque offset [Nm]; "
                             "over-speeds the rotor to create a power reserve.")
    parser.add_argument("--support-f-nom-hz", type=float, default=None,
                        help="Reference frequency [Hz] for the droop (default: "
                             "grid nominal). Set to the operating-point frequency "
                             "to isolate the event response from a shifted baseline.")
    parser.add_argument("--freq-lpf-hz", type=float, default=0.5,
                        help="Corner frequency [Hz] of the low-pass filter on the "
                             "grid-frequency measurement (0 = off).")
    parser.add_argument("--freq-lpf-order", type=int, default=2,
                        help="Number of cascaded first-order sections in the "
                             "frequency low-pass filter.")
    parser.add_argument("--support-notch-hz", type=float, default=0.0,
                        help="Centre frequency [Hz] of a band-stop (notch) on the "
                             "grid-frequency measurement, tuned to the tower mode "
                             "so the droop/inertia support does not pump the tower "
                             "resonance (0 = off).")
    parser.add_argument("--support-notch-q", type=float, default=2.0,
                        help="Quality factor of the support notch (higher = "
                             "narrower band-stop).")
    parser.add_argument("--fix-leogo-xqt", action="store_true",
                        help="Fix the LEOGO generator data artifact X_q_t<X_q_st "
                             "(non-physical) that gives a lightly damped ~5.3 Hz "
                             "mode. Raises X_q_t to --leogo-xqt.")
    parser.add_argument("--leogo-xqt", type=float, default=0.40,
                        help="Physical q-axis transient reactance for the fix.")
    parser.add_argument("--perfect-tracking", type=int, default=0, choices=[0, 1],
                        help="UIC perfect_tracking: 0 = grid disturbance reaches "
                             "the WT (default); 1 = converter holds its internal "
                             "frequency at nominal (isochronous-like), which "
                             "decouples the WT from grid-frequency drift.")
    parser.add_argument("--support-start", type=float, default=10.0,
                        help="Sim time [s] at which the support action ramps in.")
    parser.add_argument("--support-ramp-s", type=float, default=5.0)
    parser.add_argument("--support-deadband-hz", type=float, default=0.0)
    parser.add_argument("--support-max-nm", type=float, default=3.0e6,
                        help="Saturation of the torque offset [Nm].")
    parser.add_argument("--support-max-over-nm", type=float, default=None,
                        help="Asymmetric UPPER clip on the torque offset [Nm] to "
                             "cap the inertia burst so WT power stays near rating "
                             "(default: symmetric with --support-max-nm).")
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--zmq-log", type=str, default=None)
    parser.add_argument("--wt-pref-mw", type=float, default=None,
                        help="WT active-power injection [MW] assumed by the power "
                             "flow when dispatching the LEOGO gas turbines. Set to "
                             "the actual FMU operating-point power (e.g. 12.88 MW in "
                             "Region 3) so the gensets are dispatched consistently "
                             "and the steady-state grid frequency initialises at "
                             "50 Hz. Default reproduces the old ~6.6 MW Region-2 "
                             "guess.")

    args = parser.parse_args()

    t_wall_start = time.perf_counter()

    # Build combined LEOGO + FMU-turbine model
    model = build_model()

    # Optional fix for a LEOGO data-quality artifact: the synchronous generators
    # have X_q_t (0.01) < X_q_st (0.159), which is non-physical (q-axis transient
    # reactance below subtransient) and produces a very lightly damped ~5.3 Hz
    # mode that the grid-forming converter mirrors into power/frequency. Raising
    # X_q_t to a physical value (X_q > X_q_t > X_q_st) damps it.
    if args.fix_leogo_xqt:
        _gen_rows = model["generators"]["GEN"]
        _j = _gen_rows[0].index("X_q_t")
        _old = _gen_rows[1][_j]
        for _row in _gen_rows[1:]:
            _row[_j] = args.leogo_xqt
        print(f"LEOGO fix: GEN X_q_t {_old} -> {args.leogo_xqt} "
              f"(damps the ~5.3 Hz q-axis artifact mode)")

    # Optional UIC perfect_tracking: hold the converter internal frequency at
    # nominal (isochronous-like). Default 0 keeps the grid-frequency disturbance
    # coupled to the WT; 1 decouples it (converter buffers the frequency drift).
    if args.perfect_tracking != 0:
        _uic = model["vsc"]["UIC_sig"]
        _j = _uic[0].index("perfect_tracking")
        for _row in _uic[1:]:
            _row[_j] = args.perfect_tracking
        print(f"UIC perfect_tracking set to {args.perfect_tracking} "
              f"(converter holds internal frequency at nominal)")

    # Optionally force a specific OpenFAST FMU. 'fast' selects the plain
    # fast.fmu (exposes HSShftTq for the drivetrain torsion mode, faster,
    # no per-step ROSCO debug writes); 'debug'/'auto' keep fast_debug.fmu
    # (needed for the side-to-side tower output YawBrTAyp).
    if args.fmu != "auto":
        _row = model["FMUtoUICdrivetrain"]["FMUtoUICdrivetrain"]
        _hdr, _val = _row[0], _row[1]
        if args.fmu == "fast":
            _cands = [PROJECT_ROOT / "OpenFAST" / "fast.fmu",
                      PROJECT_ROOT / "fast.fmu"]
            _fname = "fast.fmu"
        else:  # debug
            _cands = [PROJECT_ROOT / "OpenFAST" / "fast_debug.fmu",
                      PROJECT_ROOT / "fast_debug.fmu"]
            _fname = "fast_debug.fmu"
        for _cand in _cands:
            if _cand.is_file():
                _val[_hdr.index("FMU_path")] = str(_cand)
                _val[_hdr.index("fmu_filename")] = _fname
                print(f"Using FMU: {_cand}")
                break
        else:
            print(f"WARNING: requested --fmu {args.fmu} not found; "
                  f"keeping default {_val[_hdr.index('FMU_path')]}")

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)

    fmu_model = ps.FMUtoUICdrivetrain["FMUtoUICdrivetrain"]
    uic_model = ps.vsc["UIC_sig"]
    gen_model = ps.gen["GEN"]

    # Optional in-process grid-driven ROSCO torque coupling. Create and start it
    # BEFORE init_dyn_sim() so the responder is already bound when the FMU is
    # primed (ROSCO may issue a ZMQ request during initialisation). The offset
    # stays zero until the first update() call inside the simulation loop.
    responder = None
    if args.zmq_grid:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rosco_zmq_grid_coupling import GridTorqueResponder
        _zmq_log = args.zmq_log or str(
            PROJECT_ROOT / "results" / "sweep" / "grid_zmq_log.csv"
        )
        responder = GridTorqueResponder(
            port=args.zmq_port,
            f_nom_hz=(args.support_f_nom_hz if args.support_f_nom_hz is not None
                      else float(np.atleast_1d(gen_model.sys_par["f_n"])[0])),
            droop_nm_per_hz=args.droop_nm_per_hz,
            inertia_nm_s_per_hz=args.inertia_nm_s_per_hz,
            deload_nm=args.deload_nm,
            support_start_s=args.support_start,
            ramp_s=args.support_ramp_s,
            deadband_hz=args.support_deadband_hz,
            max_offset_nm=args.support_max_nm,
            max_over_nm=args.support_max_over_nm,
            freq_lpf_hz=args.freq_lpf_hz,
            freq_lpf_order=args.freq_lpf_order,
            notch_hz=args.support_notch_hz,
            notch_q=args.support_notch_q,
            log_path=_zmq_log,
        )
        responder.start()

    # Initialise the UIC active-power reference for the power flow.
    # The FMU has not stepped yet, so use its rated electrical output as the
    # power-flow guess (P_rated on the WT base -> UIC base). The FMU sets the
    # true reference through the P_ref connection once the co-simulation runs.
    wt_s_n = float(fmu_model.par["S_n"][0])
    uic_s_n = float(uic_model.par["S_n"][0])
    # At 8 m/s (Region 2) the turbine produces ~6.6 MW, not the rated 15 MW.
    # The gas-turbine governors are referenced to the power-flow dispatch, so if
    # this guess does not match the FMU operating point the surplus/deficit is
    # taken up by droop and the steady-state frequency drifts off 50 Hz. Pass
    # --wt-pref-mw with the actual FMU power (e.g. 12.88 MW in Region 3) to pin
    # the operating point to 50 Hz.
    if args.wt_pref_mw is not None:
        p_ref_guess_uic_pu = args.wt_pref_mw / uic_s_n
        print(f"WT power-flow injection set to {args.wt_pref_mw:.3f} MW "
              f"({p_ref_guess_uic_pu:.4f} pu on {uic_s_n:.0f} MVA UIC base)")
    else:
        p_ref_guess_uic_pu = 0.443 * wt_s_n / uic_s_n   # ~6.6 MW / 15 MVA on WT base

    uic_model.par["p_ref"][:] = p_ref_guess_uic_pu
    uic_model.par["q_ref"][:] = 0.0

    # Power flow and dynamic initialisation.
    # init_dyn_sim() also primes the FMU (advances it 0 -> fmu_dt).
    ps.power_flow()
    ps.init_dyn_sim()

    x0 = ps.x0.copy()

    # Lock the network step to the FMU communication step.
    dt = float(fmu_model._fmu_dt)
    if abs(dt - args.dt) > 1e-12:
        print(f"Overriding --dt {args.dt} with fmu_dt {dt} (FMU requires fixed step).")

    print("\nInitialised LEOGO + OpenFAST FMU turbine")
    print(f"WT UIC name: {uic_model.par['name'][0]}")
    print(f"WT electrical bus: {uic_model.par['bus'][0]}")
    print(f"FMU communication step: {dt:.4f} s")

    # Reduced-network index of Busbar WTG1 LV (fault location).
    fault_bus_idx = uic_model.bus_idx_red["terminal"][0]

    # Reduced-network index of the main grid (gas-turbine) bus, where the
    # smooth load step is applied. This is the network-side frequency event.
    s_base_mva = float(model["base_mva"])
    load_bus_idx = gen_model.bus_idx_red["terminal"][0]
    y_load_step = (
        args.load_step_mw / s_base_mva - 1j * args.load_step_mvar / s_base_mva
    )

    def smoothstep(u):
        """Smooth 0 -> 1 transition (zero slope at both ends)."""
        u = float(np.clip(u, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def load_event_scale(t):
        """Fraction of the extra load currently applied (smooth ramps)."""
        t_on = args.event_time
        t_off = args.event_time + args.event_duration

        if t < t_on:
            return 0.0
        if args.load_ramp_on_s > 0.0 and t < t_on + args.load_ramp_on_s:
            return smoothstep((t - t_on) / args.load_ramp_on_s)
        if t < t_off:
            return 1.0
        if args.load_ramp_off_s > 0.0 and t < t_off + args.load_ramp_off_s:
            return 1.0 - smoothstep((t - t_off) / args.load_ramp_off_s)
        return 0.0

    sine_phase_rad = np.radians(args.load_sine_phase_deg)

    def load_event_sine_scale(t):
        """
        Applied load fraction including the optional sinusoidal modulation.

        The smooth step envelope gates the sinusoid on/off, so the oscillation
        only acts while the disturbance is active. With the default amplitude
        of zero this returns the plain step envelope.
        """
        envelope = load_event_scale(t)
        if envelope == 0.0:
            return 0.0
        modulation = args.load_sine_mean + args.load_sine_amplitude * np.sin(
            2.0 * np.pi * args.load_sine_freq_hz * (t - args.event_time)
            + sine_phase_rad
        )
        return envelope * modulation

    def set_load_step(t):
        ps.y_bus_red_mod[(load_bus_idx, load_bus_idx)] = (
            load_event_sine_scale(t) * y_load_step
        )

    def set_fault(t):
        fault_active = (
            args.fault
            and args.fault_start <= t <= args.fault_start + args.fault_duration
        )
        ps.y_bus_red_mod[(fault_bus_idx, fault_bus_idx)] = (
            args.fault_admittance if fault_active else 0.0
        )

    # The FMU is NOT integrated inside f_ode: state_derivatives() only reads
    # the cached FMU measurements. The FMU is advanced once per macro-step by
    # step_fmu() after the network step, matching test_WT_FMU_drivetrain_sim.py.
    def f_ode(t, x):
        set_load_step(t)
        set_fault(t)
        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(f_ode, 0.0, x0, args.t_end, max_step=dt)

    # Store t = 0 point
    set_load_step(0.0)
    set_fault(0.0)
    v0 = ps.solve_algebraic(0.0, x0)
    rows = [collect_results(ps, fmu_model, uic_model, gen_model, 0.0, x0, v0)]
    rows[0]["load_step_scale"] = load_event_sine_scale(0.0)
    rows[0]["torque_mod_factor"] = 1.0
    rows[0]["elecpwr_mod_factor"] = 1.0
    rows[0]["zmq_torque_offset_nm"] = 0.0

    torque_mod_phase = 0.0

    def torque_mod_factor(t):
        """Sinusoidal multiplier on the coupling torque command (1.0 = off)."""
        if args.torque_mod_amplitude == 0.0 or t < args.torque_mod_start:
            return 1.0
        return 1.0 + args.torque_mod_amplitude * np.sin(
            2.0 * np.pi * args.torque_mod_freq_hz * (t - args.torque_mod_start)
            + torque_mod_phase
        )

    def elecpwr_mod_factor(t):
        """Sinusoidal multiplier on the demanded electrical power (1.0 = off)."""
        if args.elecpwr_mod_amplitude == 0.0 or t < args.elecpwr_mod_start:
            return 1.0
        return 1.0 + args.elecpwr_mod_amplitude * np.sin(
            2.0 * np.pi * args.elecpwr_mod_freq_hz * (t - args.elecpwr_mod_start)
        )

    x = x0
    t = 0.0
    while solver.t < args.t_end:
        solver.step()
        x = solver.x
        t = solver.t
        v = ps.solve_algebraic(t, x)

        # Set the diagnostic modulations for this step before the FMU is
        # advanced (step_fmu reads these attributes).
        fmu_model._te_mod_factor = torque_mod_factor(t)
        fmu_model._epc_mod_factor = elecpwr_mod_factor(t)

        # Feed the live grid frequency to the in-process ZMQ responder so that
        # ROSCO - which requests a setpoint from inside step_fmu - receives a
        # generator-torque offset derived from the current LEOGO grid state.
        if responder is not None:
            responder.update(t, grid_frequency_hz(gen_model, x, v))

        # Advance the OpenFAST FMU one communication step with the latest
        # grid state, then cache its outputs for the next network step.
        fmu_model.step_fmu(x, v, t, dt)

        row = collect_results(ps, fmu_model, uic_model, gen_model, t, x, v)
        row["load_step_scale"] = load_event_sine_scale(t)
        row["torque_mod_factor"] = fmu_model._te_mod_factor
        row["elecpwr_mod_factor"] = fmu_model._epc_mod_factor
        row["zmq_torque_offset_nm"] = (
            responder.current_offset() if responder is not None else 0.0
        )
        rows.append(row)

        progress = min(100, int(100 * t / args.t_end))
        print(f"\rSimulation progress: {progress:3d}%", end="", flush=True)

    print("\rSimulation progress: 100%")

    # Always terminate the FMU to free the co-simulation instance. ROSCO issues
    # one final ZMQ request during finalisation (iStatus == -1), so the
    # in-process responder must stay alive until AFTER the FMU is terminated;
    # otherwise ROSCO's blocking REQ never gets a reply and terminate_fmu hangs.
    if hasattr(fmu_model, "terminate_fmu"):
        fmu_model.terminate_fmu()

    # Now stop the in-process ZMQ responder (writes its log).
    if responder is not None:
        responder.close()

    results = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / "casestudies" / "dyn_sim" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        output_file = Path(args.out)
        if not output_file.is_absolute():
            output_file = PROJECT_ROOT / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = output_dir / "WT1_LEOGO_FMU_results.csv"
    results.to_csv(output_file, index=False)

    print(f"Results written to: {output_file}")
    print(f"Simulation wall time: {time.perf_counter() - t_wall_start:.2f} s")

    if args.show:
        fig, ax = plt.subplots()
        ax.plot(results["t"], results["V_WTG1_LV_pu"])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("|V| at Busbar WTG1 LV [pu]")
        ax.grid(True)

        fig, ax = plt.subplots()
        ax.plot(results["t"], results["P_uic_bus_sys_pu"], label="UIC electrical power")
        if "fmu_RotSpeed" in results:
            ax2 = ax.twinx()
            ax2.plot(results["t"], results["fmu_RotSpeed"], color="tab:orange",
                     label="Rotor speed (FMU)")
            ax2.set_ylabel("Rotor speed [rpm]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Power [pu on system base]")
        ax.grid(True)
        ax.legend(loc="upper right")

        plt.show()


if __name__ == "__main__":
    main()
