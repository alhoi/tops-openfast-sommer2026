from pathlib import Path
from copy import deepcopy
import argparse
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# .../tops-openfast-sommer2026
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tops.dynamic as dps
import tops.solvers as dps_sol
import tops_openfast.dyn_models as ext_lib

from LEOGO.LEOGO_ps import load as load_leogo


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------
#
# This build extends test_WT_LEOGO_sim.py from ONE to TWO wind turbines
# so that turbine-to-turbine interaction through the LEOGO grid can be
# studied. The two turbines are electrically identical but experience
# different wind speeds (set later, in main()), which places them in
# different operating regions.
#
# WT1  -> Busbar WTG1 LV  (already wired in LEOGO)
# WT2  -> Busbar WTG2 LV  (already wired in LEOGO, previously unused)
#
# Because UIC_sig has single-unit assumptions (scalar current-limit
# check, par['xf'][0], par['S_n'][0]), each converter MUST be its own
# single-unit model object. The same is true for the turbine model.
# We therefore use the dedicated aliases:
#   UIC_sig  / UIC_sig2                   (two converter objects)
#   WindTurbineTower / WindTurbineTower2  (two turbine objects)
# Each turbine connects to its converter BY NAME (WT1->WT1_LEOGO,
# WT2->WT2_LEOGO).
#
# The turbines use the WindTurbineTower model (drivetrain torsion + tower
# side-to-side and fore-aft modal dynamics). WT1 runs with frequency-support
# droop ENABLED and WT2 with droop DISABLED, so their responses to a shared
# electrical event can be contrasted.


def build_model(
    droop_enable_wt1=1, droop_enable_wt2=0,
    k_droop_pu_per_hz=0.75, headroom_pu=0.05,
    ss_enable=1, f_ss_hz=0.234, zeta_ss=0.0034, g_ss=0.01825,
    fa_enable=1, f_fa_hz=0.235, zeta_fa=0.0125, g_fa=0.102,
    f_ss_hz_wt2=None,
):
    """
    Start from the LEOGO electrical network and place two independent
    single-unit WindTurbineTower + UIC models at Busbar WTG1 LV and
    Busbar WTG2 LV.

    WT1 runs with frequency-support droop enabled and WT2 with droop
    disabled. The tower side-to-side / fore-aft parameters use the
    calibrated defaults (zeta_ss=0.0034, g_ss=0.01825). WT2's tower
    side-to-side natural frequency can be de-tuned from WT1's via
    f_ss_hz_wt2 (defaults to f_ss_hz, i.e. identical turbines).
    """
    f_ss_hz_wt2 = f_ss_hz if f_ss_hz_wt2 is None else f_ss_hz_wt2
    model = deepcopy(load_leogo())

    wt1_bus = 'Busbar WTG1 LV'
    wt2_bus = 'Busbar WTG2 LV'

    # -------------------------------------------------------------
    # 1. Remove the original LEOGO converter at WTG1.
    # -------------------------------------------------------------
    old_converter = model['vsc'].pop('GridSideConverter_PV', None)

    if old_converter is None:
        raise KeyError(
            "Could not find model['vsc']['GridSideConverter_PV']. "
            "Check LEOGO_ps.py before running."
        )

    # -------------------------------------------------------------
    # 2. Two UIC converters, one per turbine bus.
    #    UIC_sig  -> WT1_LEOGO at Busbar WTG1 LV
    #    UIC_sig2 -> WT2_LEOGO at Busbar WTG2 LV
    # -------------------------------------------------------------
    uic_header = [
        'name', 'bus', 'S_n', 'V_n',
        'v_ref', 'p_ref', 'q_ref',
        'Ki', 'Kv', 'xf',
        'perfect_tracking', 'T_filter',
    ]

    model['vsc']['UIC_sig'] = [
        uic_header,
        [
            'WT1_LEOGO',       # UIC component name
            wt1_bus,           # LEOGO electrical connection point
            20.0,              # MVA
            0.69,              # kV: Busbar WTG1 LV is a 690 V bus
            1.05,              # V reference (matches original LEOGO converter)
            0.0,               # Set from wind speed before power flow
            0.0,
            0.03,
            0.0,
            0.1,
            0,                 # perfect_tracking=0: UIC built-in droop ACTIVE
            0.01,
        ],
    ]

    model['vsc']['UIC_sig2'] = [
        uic_header,
        [
            'WT2_LEOGO',       # UIC component name
            wt2_bus,           # LEOGO electrical connection point
            20.0,
            0.69,              # kV: Busbar WTG2 LV is a 690 V bus
            1.05,
            0.0,               # Set from wind speed before power flow
            0.0,
            0.03,
            0.0,
            0.1,
            0,
            0.01,
        ],
    ]

    # -------------------------------------------------------------
    # 3. Two WindTurbineTower models, one per converter.
    #    Both use identical machine / tower parameters; only the wind
    #    speed (main()) and the frequency-support droop differ:
    #      WT1 -> droop_enable_wt1 (default ON)
    #      WT2 -> droop_enable_wt2 (default OFF)
    #
    #    The top-level container key must equal the module name so TOPS
    #    resolves getattr(ext_lib, 'windturbine_tower').WindTurbineTower(2).
    # -------------------------------------------------------------
    wt_header = [
        'name', 'UIC', 'S_n', 'V_n',
        'J_m', 'J_e', 'K', 'D',
        'Kp_pitch', 'Ki_pitch', 'T_pitch',
        'max_pitch', 'min_pitch', 'max_pitch_rate',
        'rho', 'R', 'P_rated', 'omega_m_rated',
        'wind_rated', 'efficiency',
        'MPT_filename', 'Cp_filename',
        'speed_lpf_type', 'speed_lpf_corner_rad_s',
        'speed_lpf_damping',
        'f_nom_hz', 'droop_enable', 'K_droop_pu_per_hz', 'headroom_pu',
        'ss_enable', 'f_ss_hz', 'zeta_ss', 'g_ss',
        'fa_enable', 'f_fa_hz', 'zeta_fa', 'g_fa',
    ]

    def _wt_row(name, uic_name, droop_enable, f_ss):
        return [
            name,
            uic_name,        # Must match a UIC name above
            15.0,
            0.69,
            352460500.0,
            1836784.0,
            69737644900.0 / 100.0,
            35698200.0 / 10.0,
            0.6738,
            0.06,
            2.2,
            30.0,
            0.0,
            10.0,
            1.225,
            120.97,
            1.0,
            7.559987120819503,
            10.6,
            0.95756,
            'MPT_Kopt2150.csv',
            'Cp_Ct_Cq.IEA15MW.ROSCO.txt',
            2,
            1.00810,
            0.70000,
            50.0,
            int(droop_enable), float(k_droop_pu_per_hz), float(headroom_pu),
            int(ss_enable), float(f_ss), float(zeta_ss), float(g_ss),
            int(fa_enable), float(f_fa_hz), float(zeta_fa), float(g_fa),
        ]

    model['windturbine_tower'] = {
        'WindTurbineTower': [
            wt_header, _wt_row('WT1', 'WT1_LEOGO', droop_enable_wt1, f_ss_hz),
        ],
        'WindTurbineTower2': [
            wt_header,
            _wt_row('WT2', 'WT2_LEOGO', droop_enable_wt2, f_ss_hz_wt2),
        ],
    }

    return model


# ---------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------

def scalar(value):
    """Return a scalar float from a scalar or length-one NumPy array."""
    return float(np.asarray(value).reshape(-1)[0])


def set_constant_wind(wt_model, wind_mps):
    """
    Force a turbine to see a constant wind speed.

    The base WindTurbine.wind_speed / wind_speed_init return a hardcoded
    value. Assigning instance-attribute lambdas shadows those class
    methods so each turbine can be given its own wind, before power flow
    and dynamic initialisation read it.
    """
    wt_model.wind_speed = lambda x, v, u=float(wind_mps): u
    wt_model.wind_speed_init = lambda u=float(wind_mps): u


def set_wind_with_gust(wt_model, base_mps, gust_start, gust_duration,
                       gust_delta_mps):
    """
    Give a turbine a constant base wind plus a smooth transient gust.

    The gust is a raised-cosine bump (0 -> peak -> 0) of height
    gust_delta_mps, starting at gust_start and lasting gust_duration. The
    installed wind_speed lambda reads the turbine's _sim_time (set each step
    by the driver), so wind_speed(x, v) returns the instantaneous wind during
    the dynamic simulation. Dynamic initialisation uses the base wind only.
    """
    u0 = float(base_mps)
    t0 = float(gust_start)
    dur = float(gust_duration)
    d_u = float(gust_delta_mps)

    def gust_shape(t):
        if dur <= 0.0 or t < t0 or t > t0 + dur:
            return 0.0
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - t0) / dur))

    def wind_fn(x, v, wt=wt_model):
        t = float(getattr(wt, '_sim_time', 0.0))
        return u0 + d_u * gust_shape(t)

    wt_model.wind_speed = wind_fn
    wt_model.wind_speed_init = lambda u=u0: u


def apply_pref_step(wt_model, step_uic_pu, start_s, ramp_s=0.1):
    """
    Add a smooth, held step to a turbine's active-power reference.

    This wraps the turbine's P_ref(x, v) output (fed to its converter each
    step by the connections mechanism) so the converter receives a localized
    electrical setpoint change on one turbine only. With no other disturbance,
    the turbine's own mechanical response isolates the electrical -> mechanical
    coupling inside one turbine, while the other turbine's response isolates
    the turbine-to-turbine coupling through the grid.
    """
    orig_p_ref = wt_model.P_ref
    step = float(step_uic_pu)
    t0 = float(start_s)
    ramp = float(ramp_s)

    def p_ref_stepped(x, v):
        base = np.atleast_1d(orig_p_ref(x, v)).astype(float)
        t = float(getattr(wt_model, '_sim_time', 0.0))
        if t < t0:
            scale = 0.0
        elif ramp > 0.0 and t < t0 + ramp:
            u = (t - t0) / ramp
            scale = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        else:
            scale = 1.0
        return base + scale * step

    wt_model.P_ref = p_ref_stepped


def _turbine_signals(prefix, wt_model, uic_model, x, v, sys_s_n):
    """Return one turbine's key signals, keyed with a per-turbine suffix."""
    wt_s_n = float(wt_model.par['S_n'][0])
    uic_s_n = float(uic_model.par['S_n'][0])

    wt_states = wt_model.local_view(x)
    omega_m_pu = scalar(wt_states['omega_m'])
    omega_e_pu = scalar(wt_states['omega_e'])
    pitch_rad = (
        scalar(wt_states['pitch_angle'])
        if 'pitch_angle' in wt_states.dtype.names
        else 0.0
    )

    # Tower modal states + acceleration outputs (side-to-side, fore-aft)
    q_ss = scalar(wt_states['q_ss']) if 'q_ss' in wt_states.dtype.names else 0.0
    q_fa = scalar(wt_states['q_fa']) if 'q_fa' in wt_states.dtype.names else 0.0
    ss_accel = scalar(wt_model.ss_acceleration(x, v))
    fa_accel = scalar(wt_model.fa_acceleration(x, v))

    v_terminal = uic_model.v_t(x, v)[0]
    s_uic = uic_model.s_e(x, v)[0]
    i_a = uic_model.i_a(x, v)[0]

    uic_states = uic_model.local_view(x)
    vi = uic_states['vi_x'][0] + 1j * uic_states['vi_y'][0]

    return {
        f'wind_speed_mps_{prefix}': scalar(wt_model.wind_speed(x, v)),
        f'omega_m_pu_{prefix}': omega_m_pu,
        f'omega_e_pu_{prefix}': omega_e_pu,
        f'pitch_deg_{prefix}': np.degrees(pitch_rad),
        f'P_aero_sys_pu_{prefix}': scalar(wt_model.P_aero(x, v)) * wt_s_n / sys_s_n,
        f'P_e_sys_pu_{prefix}': scalar(wt_model.P_e(x, v)) * uic_s_n / sys_s_n,
        f'P_ref_sys_pu_{prefix}': scalar(wt_model.P_ref(x, v)) * uic_s_n / sys_s_n,
        f'ss_accel_mps2_{prefix}': ss_accel,
        f'fa_accel_mps2_{prefix}': fa_accel,
        f'q_ss_{prefix}': q_ss,
        f'q_fa_{prefix}': q_fa,
        f'V_LV_pu_{prefix}': abs(v_terminal),
        f'angle_LV_deg_{prefix}': np.degrees(np.angle(v_terminal)),
        f'V_uic_internal_pu_{prefix}': abs(vi),
        f'P_uic_bus_sys_pu_{prefix}': s_uic.real * uic_s_n / sys_s_n,
        f'Q_uic_bus_sys_pu_{prefix}': s_uic.imag * uic_s_n / sys_s_n,
        f'I_uic_pu_{prefix}': abs(i_a),
    }


def collect_results(ps, wt1_model, uic1_model, wt2_model, uic2_model,
                    gen_model, t, x, v, coi0=0.0):
    """
    Store key signals for both turbines on the LEOGO system base, plus
    total synchronous generation and the grid (centre-of-inertia)
    frequency shared by both turbines.
    """
    sys_s_n = float(wt1_model.sys_par['s_n'])

    row = {'t': t}
    row.update(_turbine_signals('wt1', wt1_model, uic1_model, x, v, sys_s_n))
    row.update(_turbine_signals('wt2', wt2_model, uic2_model, x, v, sys_s_n))

    # Total synchronous-generator electrical output
    p_gen_local = np.asarray(gen_model.p_e(x, v), dtype=float)
    q_gen_local = np.asarray(gen_model.q_e(x, v), dtype=float)
    gen_s_n = np.asarray(gen_model.par['S_n'], dtype=float)
    row['P_sync_generators_total_sys_pu'] = float(
        np.sum(p_gen_local * gen_s_n / sys_s_n)
    )
    row['Q_sync_generators_total_sys_pu'] = float(
        np.sum(q_gen_local * gen_s_n / sys_s_n)
    )

    # Grid frequency from the synchronous machines: f = f_n * (1 + speed_pu).
    f_n = float(gen_model.sys_par['f_n'])
    gen_speed_pu = np.asarray(gen_model.speed(x, v), dtype=float).ravel()
    # Referenced to the settled operating point coi0 (see warm-up in main): the
    # baseline then reads f_n exactly and the plot shows the true deviation.
    row['grid_freq_hz'] = f_n * (1.0 + float(np.mean(gen_speed_pu)) - coi0)

    return row


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-end', type=float, default=200.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--warmup-s', type=float, default=0.0,
                        help='Relax the model this many seconds before t=0 so '
                             'the pre-event baseline is settled (0=off).')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--out', type=str, default=None,
                        help='Output CSV path (default WT_LEOGO_2wt_results.csv '
                             'in the results directory).')

    # Different wind for the two turbines. Defaults put WT1 in Region 3
    # (above rated ~10.6 m/s, pitch-controlled) and WT2 in Region 2
    # (below rated, torque/MPPT-controlled).
    parser.add_argument('--wind-wt1', type=float, default=12.0,
                        help='Constant wind speed for WT1 [m/s].')
    parser.add_argument('--wind-wt2', type=float, default=9.0,
                        help='Constant wind speed for WT2 [m/s].')

    # Optional smooth wind gust on WT1 (isolated mechanical excitation of one
    # turbine, on top of its constant base wind). Raised-cosine bump.
    parser.add_argument('--wt1-gust-delta', type=float, default=0.0,
                        help='WT1 wind-gust peak amplitude [m/s] (0=off).')
    parser.add_argument('--wt1-gust-start', type=float, default=0.0,
                        help='WT1 wind-gust start time [s].')
    parser.add_argument('--wt1-gust-duration', type=float, default=6.0,
                        help='WT1 wind-gust duration [s].')

    # Optional localized electrical perturbation: a smooth held step on WT1's
    # power reference only (no network change). Isolates E->M within WT1 (its
    # own response) and WT1->WT2 coupling through the grid (WT2's response).
    parser.add_argument('--wt1-pref-step', type=float, default=0.0,
                        help='WT1 P_ref step [pu on UIC base] (0=off).')
    parser.add_argument('--wt1-pref-start', type=float, default=10.0,
                        help='WT1 P_ref step start time [s].')

    # Optional resonant electrical excitation: a small cyclic active-power load
    # fluctuation on the main grid bus, tuned to a tower natural frequency. The
    # LEOGO spec notes a ~+/-4% demand fluctuation (~+/-2 MW on ~48 MW), so a
    # few-MW cyclic load is realistic; aligning it with a tower mode probes the
    # electro-mechanical coupling (grid -> WT1 frequency droop -> torque/thrust
    # -> tower) at resonance.
    parser.add_argument('--osc-load-mw', type=float, default=0.0,
                        help='Peak of the cyclic active-power load fluctuation '
                             'on the main grid bus [MW] (0=off).')
    parser.add_argument('--osc-load-mvar', type=float, default=0.0,
                        help='Peak of the cyclic reactive-power load '
                             'fluctuation on the main grid bus [Mvar].')
    parser.add_argument('--osc-freq-hz', type=float, default=0.234,
                        help='Frequency of the cyclic load fluctuation [Hz]; '
                             'set to a tower natural frequency to probe '
                             'resonance.')
    parser.add_argument('--osc-start', type=float, default=10.0,
                        help='Start time of the cyclic load fluctuation [s].')
    parser.add_argument('--osc-duration', type=float, default=0.0,
                        help='Duration of the cyclic load fluctuation [s] '
                             '(0=until t_end).')

    # Smooth load step applied as a temporary shunt at the main grid bus.
    parser.add_argument('--load-step-mw', type=float, default=0.0)
    parser.add_argument('--load-step-mvar', type=float, default=0.0)
    parser.add_argument('--event-time', type=float, default=10.0)
    parser.add_argument('--event-duration', type=float, default=30.0)
    parser.add_argument(
        '--load-ramp-on-s',
        type=float,
        default=2.0,
        help='Duration of the smooth load increase. Zero gives an ideal step.',
    )
    parser.add_argument(
        '--load-ramp-off-s',
        type=float,
        default=2.0,
        help='Duration of the smooth load removal. Zero gives an ideal step.',
    )

    # Under-frequency load shedding (GT-trip scheme): a permanent load
    # reduction switched in a short delay after the load step.
    parser.add_argument('--load-shed-mw', type=float, default=0.0,
                        help='Under-frequency load shed [MW] (0=off).')
    parser.add_argument('--load-shed-delay-s', type=float, default=0.2,
                        help='Delay after event-time before shedding [s].')

    # Optional three-phase fault at the WT1 connection point
    parser.add_argument('--fault', action='store_true')
    parser.add_argument('--fault-start', type=float, default=3.0)
    parser.add_argument('--fault-duration', type=float, default=0.05)
    parser.add_argument('--fault-admittance', type=float, default=1e6)

    # Frequency-support droop: ON for WT1, OFF for WT2 by default.
    parser.add_argument('--droop-wt1', type=int, default=1,
                        help='WT1 frequency-support droop (1=on, 0=off).')
    parser.add_argument('--droop-wt2', type=int, default=0,
                        help='WT2 frequency-support droop (1=on, 0=off).')
    parser.add_argument('--k-droop', type=float, default=0.75,
                        help='Droop gain [pu power per Hz].')
    parser.add_argument('--headroom', type=float, default=0.05,
                        help='De-loaded reserve for droop [pu on UIC base].')

    # Tower side-to-side / fore-aft modal parameters (calibrated defaults).
    parser.add_argument('--ss-enable', type=int, default=1)
    parser.add_argument('--f-ss-hz', type=float, default=0.234)
    parser.add_argument('--f-ss-hz-wt2', type=float, default=None,
                        help='WT2 tower side-to-side natural frequency [Hz] '
                             '(default: same as --f-ss-hz).')
    parser.add_argument('--zeta-ss', type=float, default=0.0034)
    parser.add_argument('--g-ss', type=float, default=0.01825)
    parser.add_argument('--fa-enable', type=int, default=1)
    parser.add_argument('--f-fa-hz', type=float, default=0.235)
    parser.add_argument('--zeta-fa', type=float, default=0.0125)
    parser.add_argument('--g-fa', type=float, default=0.102)

    # Optional reciprocal tower-SS <-> drivetrain feedback (default OFF).
    parser.add_argument('--ss-feedback', action='store_true',
                        help='Enable the two-way tower-SS <-> drivetrain '
                             'coupling (default off = forward-coupled only).')
    parser.add_argument('--ss-feedback-c', type=float, default=0.05,
                        help='SS<->drivetrain inertia coupling coefficient c '
                             '(needs c**2 < 2*H_e ~= 0.077); larger = stronger '
                             'feedback.')
    parser.add_argument('--ss-feedback-target', choices=['wt1', 'wt2', 'both'],
                        default='wt1',
                        help='Which turbine(s) receive the two-way coupling.')

    args = parser.parse_args()

    t_wall_start = time.perf_counter()

    # Build combined LEOGO + 2x WindTurbineTower/UIC model
    model = build_model(
        droop_enable_wt1=args.droop_wt1,
        droop_enable_wt2=args.droop_wt2,
        k_droop_pu_per_hz=args.k_droop,
        headroom_pu=args.headroom,
        ss_enable=args.ss_enable, f_ss_hz=args.f_ss_hz,
        f_ss_hz_wt2=args.f_ss_hz_wt2,
        zeta_ss=args.zeta_ss, g_ss=args.g_ss,
        fa_enable=args.fa_enable, f_fa_hz=args.f_fa_hz,
        zeta_fa=args.zeta_fa, g_fa=args.g_fa,
    )

    ps = dps.PowerSystemModel(
        model=model,
        user_mdl_lib=ext_lib,
    )

    wt1_model = ps.windturbine_tower['WindTurbineTower']
    wt2_model = ps.windturbine_tower['WindTurbineTower2']
    uic1_model = ps.vsc['UIC_sig']
    uic2_model = ps.vsc['UIC_sig2']
    gen_model = ps.gen['GEN']

    # Apply optional reciprocal tower-SS <-> drivetrain feedback (default off).
    if args.ss_feedback:
        if args.ss_feedback_target in ('wt1', 'both'):
            wt1_model.set_ss_feedback(True, args.ss_feedback_c)
        if args.ss_feedback_target in ('wt2', 'both'):
            wt2_model.set_ss_feedback(True, args.ss_feedback_c)
        print(f"Tower-SS two-way feedback ON ({args.ss_feedback_target}), "
              f"c={args.ss_feedback_c:g}")

    # -------------------------------------------------------------
    # Assign each turbine its own constant wind BEFORE power flow and
    # dynamic initialisation, so each initialises at its own operating
    # point (Region 2 vs Region 3).
    # -------------------------------------------------------------
    if args.wt1_gust_delta != 0.0:
        set_wind_with_gust(wt1_model, args.wind_wt1, args.wt1_gust_start,
                           args.wt1_gust_duration, args.wt1_gust_delta)
    else:
        set_constant_wind(wt1_model, args.wind_wt1)
    set_constant_wind(wt2_model, args.wind_wt2)

    # Initialise each converter active-power reference from its turbine's
    # MPT curve at that turbine's wind speed.
    p_ref1 = wt1_model.P_ref_from_wind(
        wt1_model.wind_speed_init(), uic1_model.par['S_n']
    )
    p_ref2 = wt2_model.P_ref_from_wind(
        wt2_model.wind_speed_init(), uic2_model.par['S_n']
    )

    uic1_model.par['p_ref'][:] = p_ref1
    uic1_model.par['q_ref'][:] = 0.0
    uic2_model.par['p_ref'][:] = p_ref2
    uic2_model.par['q_ref'][:] = 0.0

    # Power flow and dynamic initialisation
    ps.power_flow()
    ps.init_dyn_sim()

    x0 = ps.x0.copy()

    print('\nInitialised LEOGO + two-WT model')
    print(f"WT1: UIC {uic1_model.par['name'][0]} at {uic1_model.par['bus'][0]}, "
          f"wind {scalar(wt1_model.wind_speed_init()):.3f} m/s, "
          f"P_ref {scalar(p_ref1):.5f} pu (UIC base), "
          f"droop {'ON' if args.droop_wt1 else 'OFF'}")
    print(f"WT2: UIC {uic2_model.par['name'][0]} at {uic2_model.par['bus'][0]}, "
          f"wind {scalar(wt2_model.wind_speed_init()):.3f} m/s, "
          f"P_ref {scalar(p_ref2):.5f} pu (UIC base), "
          f"droop {'ON' if args.droop_wt2 else 'OFF'}")

    if args.load_step_mw != 0.0:
        msg = (f"Grid event: +{args.load_step_mw:.2f} MW load step at "
               f"t={args.event_time:.1f} s")
        if args.load_shed_mw != 0.0:
            net_mw = args.load_step_mw - args.load_shed_mw
            msg += (f", -{args.load_shed_mw:.2f} MW shed at "
                    f"t={args.event_time + args.load_shed_delay_s:.2f} s "
                    f"(net {net_mw:+.2f} MW, GT-trip scheme)")
        print(msg)
    if args.wt1_gust_delta != 0.0:
        print(f"WT1 wind gust: {args.wt1_gust_delta:+.2f} m/s peak over "
              f"{args.wt1_gust_duration:.1f} s from t={args.wt1_gust_start:.1f} s")
    if args.osc_load_mw != 0.0 or args.osc_load_mvar != 0.0:
        dur = ('until t_end' if args.osc_duration == 0.0
               else f"{args.osc_duration:.1f} s")
        print(f"Cyclic load fluctuation: {args.osc_load_mw:.2f} MW / "
              f"{args.osc_load_mvar:.2f} Mvar peak at "
              f"{args.osc_freq_hz:.3f} Hz from t={args.osc_start:.1f} s ({dur})")

    # Localized electrical perturbation on WT1 only (isolation probe). Must be
    # installed after init_dyn_sim; P_ref is read dynamically during the run.
    if args.wt1_pref_step != 0.0:
        apply_pref_step(wt1_model, args.wt1_pref_step, args.wt1_pref_start)
        print(f"WT1 P_ref step: {args.wt1_pref_step:+.3f} pu (UIC) held from "
              f"t={args.wt1_pref_start:.1f} s (isolation probe)")

    # Reduced-network indices.
    # Load step is applied at the main grid (gas-turbine) bus; the fault,
    # if enabled, is applied at the WT1 connection point.
    s_base_mva = float(model['base_mva'])
    load_bus_idx = gen_model.bus_idx_red['terminal'][0]
    fault_bus_idx = uic1_model.bus_idx_red['terminal'][0]

    y_load_step = (
        args.load_step_mw / s_base_mva - 1j * args.load_step_mvar / s_base_mva
    )

    # Under-frequency load shedding: permanent real-power load reduction at the
    # main grid bus, switched in a short delay after the load step (GT-trip
    # scheme). Enters set_load_step with a minus sign.
    y_load_shed = args.load_shed_mw / s_base_mva
    load_shed_ramp_s = 0.02

    # Cyclic (sinusoidal) load fluctuation on the main grid bus. Zero-mean, so
    # it excites the tower mode without shifting the steady operating point.
    y_osc_amp = (
        args.osc_load_mw / s_base_mva - 1j * args.osc_load_mvar / s_base_mva
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

    def load_shed_scale(t):
        """Fraction of the shed load currently removed (0 before, 1 after).

        Under-frequency load shedding is a fast breaker action, so a short
        ramp is used and the shed load then stays off for the rest of the run.
        """
        if args.load_shed_mw == 0.0:
            return 0.0
        t_shed = args.event_time + args.load_shed_delay_s
        if t < t_shed:
            return 0.0
        if load_shed_ramp_s > 0.0 and t < t_shed + load_shed_ramp_s:
            return smoothstep((t - t_shed) / load_shed_ramp_s)
        return 1.0

    def osc_load_admittance(t):
        """Complex shunt admittance of the cyclic main-bus load fluctuation.

        A zero-mean sine at osc_freq_hz, smoothly ramped in (and out) over one
        period so its time derivative stays continuous. Tuned to a tower natural
        frequency, it resonantly excites the tower mode through the grid.
        """
        if args.osc_load_mw == 0.0 and args.osc_load_mvar == 0.0:
            return 0.0
        t0 = args.osc_start
        if t < t0:
            return 0.0
        if args.osc_duration > 0.0 and t > t0 + args.osc_duration:
            return 0.0
        period = 1.0 / args.osc_freq_hz if args.osc_freq_hz > 0.0 else 0.0
        env = 1.0
        if period > 0.0 and (t - t0) < period:
            env = smoothstep((t - t0) / period)
        if args.osc_duration > 0.0 and period > 0.0:
            t_left = (t0 + args.osc_duration) - t
            if t_left < period:
                env *= smoothstep(t_left / period)
        phase = 2.0 * np.pi * args.osc_freq_hz * (t - t0)
        return env * y_osc_amp * np.sin(phase)

    def set_load_step(t):
        # Main grid-bus disturbance: smooth load step minus a permanent
        # under-frequency load shed (GT-trip scheme), plus an optional cyclic
        # load fluctuation. All act on the same bus, so they sum; the shed
        # enters with a minus sign.
        ps.y_bus_red_mod[(load_bus_idx, load_bus_idx)] = (
            load_event_scale(t) * y_load_step
            - load_shed_scale(t) * y_load_shed
            + osc_load_admittance(t)
        )

    def set_fault(t):
        """Apply or clear the optional fault at Busbar WTG1 LV."""
        fault_active = (
            args.fault
            and args.fault_start <= t <= args.fault_start + args.fault_duration
        )
        ps.y_bus_red_mod[(fault_bus_idx, fault_bus_idx)] = (
            args.fault_admittance if fault_active else 0.0
        )

    # Main-busbar frequency fed to each WT frequency-support droop. The GT
    # speed is a state, so it is available from x before the algebraic solve;
    # each WT droop reads self._grid_frequency_hz when forming P_ref.
    f_n_grid = float(np.asarray(gen_model.sys_par['f_n']).ravel()[0])

    def feed_grid_frequency(x):
        gen_speed_pu = np.asarray(gen_model.speed(x, None), dtype=float).ravel()
        f_grid = f_n_grid * (1.0 + float(np.mean(gen_speed_pu)))
        wt1_model.set_grid_frequency_hz(f_grid)
        wt2_model.set_grid_frequency_hz(f_grid)

    def f_ode(t, x):
        # Both turbines use this time internally.
        wt1_model._sim_time = t
        wt2_model._sim_time = t

        set_load_step(t)
        set_fault(t)

        # Feed the measured grid frequency to the WT droops before P_ref forms.
        feed_grid_frequency(x)

        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    # Optional warm-up: integrate the coupled model to its settled equilibrium
    # before t=0 so the pre-event baseline is flat. Every disturbance (load
    # step, fault, P_ref step, gust) triggers at t >= 0, so integrating from
    # -warmup_s to 0 keeps them all inactive during warm-up. The reduced
    # turbine models start off-equilibrium (WT1 pitch/rotor relaxes, WT2 rotor
    # creeps to its TSR optimum); warm-up absorbs that initial-condition
    # transient instead of letting it ride through the recorded window.
    if args.warmup_s > 0.0:
        warm_solver = dps_sol.SimpleRK4(
            f_ode, -args.warmup_s, x0, 0.0, max_step=args.dt,
        )
        while warm_solver.t < 0.0:
            warm_solver.step()
        x0 = warm_solver.x.copy()
        print(f"Warm-up: relaxed {args.warmup_s:.1f} s to a settled baseline "
              f"before t=0")

    # Reference the reported grid frequency to the settled operating point, so
    # the baseline reads exactly f_n. Without warm-up the generators start at
    # nominal speed (coi0 = 0), so this leaves the old behaviour unchanged.
    coi0 = float(
        np.mean(np.asarray(gen_model.speed(x0, None), dtype=float).ravel())
    )

    # After warm-up the settled torque/thrust differ from the initial guess, so
    # the tower modes still sit slightly off their new equilibrium and would
    # ring down through the recorded window. Reset both turbines' tower modes to
    # the static equilibrium of the settled operating point. The modes are
    # forward-coupled, so this does not perturb the electrical/drivetrain
    # solution: the pre-event tower baseline becomes flat and the event response
    # is the clean, event-proportional ring.
    if args.warmup_s > 0.0:
        wt1_model._sim_time = 0.0
        wt2_model._sim_time = 0.0
        set_load_step(0.0)
        set_fault(0.0)
        feed_grid_frequency(x0)
        v_settled = ps.solve_algebraic(0.0, x0)
        wt1_model.reset_tower_modes(x0, v_settled)
        wt2_model.reset_tower_modes(x0, v_settled)
        print("Tower modes reset to the settled equilibrium (flat baseline)")

    solver = dps_sol.SimpleRK4(
        f_ode,
        0.0,
        x0,
        args.t_end,
        max_step=args.dt,
    )

    # Store t = 0 point
    wt1_model._sim_time = 0.0
    wt2_model._sim_time = 0.0
    set_load_step(0.0)
    set_fault(0.0)
    feed_grid_frequency(x0)
    v0 = ps.solve_algebraic(0.0, x0)

    rows = [
        collect_results(
            ps, wt1_model, uic1_model, wt2_model, uic2_model, gen_model,
            t=0.0, x=x0, v=v0, coi0=coi0,
        )
    ]
    rows[0]['load_step_scale'] = load_event_scale(0.0)

    while solver.t < args.t_end:
        solver.step()

        t = solver.t
        x = solver.x

        wt1_model._sim_time = t
        wt2_model._sim_time = t
        set_load_step(t)
        set_fault(t)
        feed_grid_frequency(x)
        v = ps.solve_algebraic(t, x)

        row = collect_results(
            ps, wt1_model, uic1_model, wt2_model, uic2_model, gen_model,
            t=t, x=x, v=v, coi0=coi0,
        )
        row['load_step_scale'] = load_event_scale(t)
        rows.append(row)

        progress = min(100, int(100 * t / args.t_end))
        print(f'\rSimulation progress: {progress:3d}%', end='', flush=True)

    print('\rSimulation progress: 100%')

    results = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / 'casestudies' / 'dyn_sim' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        Path(args.out) if args.out else output_dir / 'WT_LEOGO_2wt_results.csv'
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print(f'Results written to: {output_file}')
    print(f'Simulation wall time: {time.perf_counter() - t_wall_start:.2f} s')

    if args.show:
        fig, ax = plt.subplots()
        ax.plot(results['t'], results['P_uic_bus_sys_pu_wt1'], label='WT1 elektrisk effekt')
        ax.plot(results['t'], results['P_uic_bus_sys_pu_wt2'], label='WT2 elektrisk effekt')
        ax.set_xlabel('Tid [s]')
        ax.set_ylabel('Effekt [pu på systembasis]')
        ax.grid(True)
        ax.legend()

        fig, ax = plt.subplots()
        ax.plot(results['t'], results['omega_m_pu_wt1'], label='WT1 rotorhastighet')
        ax.plot(results['t'], results['omega_m_pu_wt2'], label='WT2 rotorhastighet')
        ax.set_xlabel('Tid [s]')
        ax.set_ylabel('Mekanisk hastighet [pu]')
        ax.grid(True)
        ax.legend()

        plt.show()


if __name__ == '__main__':
    main()
