"""
LEOGO + simplified wind-turbine simulation using WindTurbineTower.

This is a copy of test_WT_LEOGO_sim.py wired to the WindTurbineTower model
(a superset of WindTurbine that adds a toggleable tower side-to-side (SS)
structural mode driven by the generator electromagnetic torque). The SS mode
can be switched on/off and tuned from the command line, and the tower-top
side-to-side acceleration and modal states are logged.

When the SS mode is disabled (--ss-off), the electrical/drivetrain results are
identical to test_WT_LEOGO_sim.py.
"""

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

def build_model(ss_enable=1, f_ss_hz=0.234, zeta_ss=0.0034, g_ss=0.01825,
                fa_enable=1, f_fa_hz=0.235, zeta_fa=0.0125, g_fa=0.102,
                droop_enable=0, k_droop_pu_per_hz=0.75, headroom_pu=0.05):
    """
    Start from the LEOGO electrical network and replace its original,
    simplified WT1 grid-side converter with the WindTurbineTower + UIC_sig
    model.

    The four SS parameters (ss_enable, f_ss_hz, zeta_ss, g_ss) drive the
    optional tower side-to-side modal DOF, and the four FA parameters
    (fa_enable, f_fa_hz, zeta_fa, g_fa) drive the optional tower fore-aft modal
    DOF; ss_enable=0 and fa_enable=0 reproduce the plain WindTurbine behaviour.

    The three droop parameters (droop_enable, k_droop_pu_per_hz, headroom_pu)
    drive the WindTurbineTower's own frequency-support droop. When enabled and
    fed the measured grid frequency (see set_grid_frequency_hz in the main
    loop), the turbine raises its power reference on a frequency dip, using up
    to headroom_pu of de-loaded reserve. headroom_pu de-loads the turbine even
    when droop is off, so ON/OFF runs share the same operating point and the
    difference isolates the droop *response*.
    """
    model = deepcopy(load_leogo())

    wt_bus = 'Busbar WTG1 LV'

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
    # 2. Add the UIC model at the same physical electrical bus.
    # -------------------------------------------------------------
    model['vsc']['UIC_sig'] = [
        [
            'name', 'bus', 'S_n', 'V_n',
            'v_ref', 'p_ref', 'q_ref',
            'Ki', 'Kv', 'xf',
            'perfect_tracking', 'T_filter'
        ],
        [
            'WT1_LEOGO',       # UIC component name
            wt_bus,            # LEOGO electrical connection point
            20.0,              # MVA
            0.69,              # kV: Busbar WTG1 LV is a 690 V bus
            1.05,              # Matches original LEOGO converter V reference
            0.0,               # Set from wind speed before power flow
            0.0,
            0.03,
            0.0,
            0.1,
            0,   # perfect_tracking: 0 = UIC's built-in (grid-forming) droop ACTIVE
            0.01,
        ],
    ]

    # -------------------------------------------------------------
    # 3. Add the aerodynamic WT model (WindTurbineTower).
    #
    # WT1 connects to the UIC component called WT1_LEOGO.
    #
    # NOTE: the top-level container key must match the module name that holds
    # the class (TOPS resolves getattr(ext_lib, 'windturbine_tower').
    # WindTurbineTower), hence 'windturbine_tower' rather than 'windturbine'.
    # -------------------------------------------------------------
    model['windturbine_tower'] = {
        'WindTurbineTower': [
            [
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
            ],  # noqa: E501
            [
                'WT1',
                'WT1_LEOGO',     # Must match UIC_sig name above
                15.0,
                0.69,            # Connection voltage at Busbar WTG1 LV
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
                int(droop_enable), float(k_droop_pu_per_hz), float(headroom_pu),  # WT frequency-support droop
                int(ss_enable), float(f_ss_hz), float(zeta_ss), float(g_ss),  # SS: enable, f_ss[Hz], zeta, g_ss[m/s^2/pu]
                int(fa_enable), float(f_fa_hz), float(zeta_fa), float(g_fa),  # FA: enable, f_fa[Hz], zeta, g_fa[m/s^2/pu]
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


def collect_results(ps, wt_model, uic_model, gen_model, t, x, v):
    """
    Store key signals on the LEOGO system base.
    """
    sys_s_n = float(wt_model.sys_par['s_n'])
    wt_s_n = float(wt_model.par['S_n'][0])
    uic_s_n = float(uic_model.par['S_n'][0])

    wt_states = wt_model.local_view(x)

    # WT mechanical / aerodynamic variables
    omega_m_pu = scalar(wt_states['omega_m'])
    omega_e_pu = scalar(wt_states['omega_e'])

    pitch_rad = (
        scalar(wt_states['pitch_angle'])
        if 'pitch_angle' in wt_states.dtype.names
        else 0.0
    )

    # Tower side-to-side modal states + acceleration output
    q_ss = (
        scalar(wt_states['q_ss'])
        if 'q_ss' in wt_states.dtype.names
        else 0.0
    )
    q_ss_dot = (
        scalar(wt_states['q_ss_dot'])
        if 'q_ss_dot' in wt_states.dtype.names
        else 0.0
    )
    ss_accel = scalar(wt_model.ss_acceleration(x, v))

    # Tower fore-aft modal states + acceleration output
    q_fa = (
        scalar(wt_states['q_fa'])
        if 'q_fa' in wt_states.dtype.names
        else 0.0
    )
    q_fa_dot = (
        scalar(wt_states['q_fa_dot'])
        if 'q_fa_dot' in wt_states.dtype.names
        else 0.0
    )
    fa_accel = scalar(wt_model.fa_acceleration(x, v))

    # UIC terminal variables
    v_terminal = uic_model.v_t(x, v)[0]
    s_uic = uic_model.s_e(x, v)[0]
    i_a = uic_model.i_a(x, v)[0]

    # Internal UIC voltage
    uic_states = uic_model.local_view(x)
    vi = uic_states['vi_x'][0] + 1j * uic_states['vi_y'][0]

    # Total synchronous-generator electrical output
    p_gen_local = np.asarray(gen_model.p_e(x, v), dtype=float)
    q_gen_local = np.asarray(gen_model.q_e(x, v), dtype=float)
    gen_s_n = np.asarray(gen_model.par['S_n'], dtype=float)

    p_gen_total_sys_pu = float(np.sum(p_gen_local * gen_s_n / sys_s_n))
    q_gen_total_sys_pu = float(np.sum(q_gen_local * gen_s_n / sys_s_n))

    # Grid frequency from the synchronous machines: f = f_n * (1 + speed_pu).
    # The three GTs are coherent on Main Bus A, so their mean speed gives the
    # main-busbar frequency (the quantity the WT droop/UIC reacts to).
    f_n = float(np.asarray(gen_model.sys_par['f_n']).ravel()[0])
    gen_speed_pu = np.asarray(gen_model.speed(x, v), dtype=float).ravel()
    grid_freq_hz = f_n * (1.0 + float(np.mean(gen_speed_pu)))

    return {
        't': t,

        # Main-busbar grid frequency (synchronous-machine speed)
        'grid_freq_hz': grid_freq_hz,

        # Wind turbine
        'wind_speed_mps': scalar(wt_model.wind_speed(x, v)),
        'omega_m_pu': omega_m_pu,
        'omega_e_pu': omega_e_pu,
        'pitch_deg': np.degrees(pitch_rad),
        'P_aero_sys_pu': scalar(wt_model.P_aero(x, v)) * wt_s_n / sys_s_n,
        'P_e_sys_pu': scalar(wt_model.P_e(x, v)) * uic_s_n / sys_s_n,
        'P_ref_sys_pu': scalar(wt_model.P_ref(x, v)) * uic_s_n / sys_s_n,

        # Tower side-to-side mode
        'ss_accel_mps2': ss_accel,
        'q_ss': q_ss,
        'q_ss_dot': q_ss_dot,

        # Tower fore-aft mode
        'fa_accel_mps2': fa_accel,
        'q_fa': q_fa,
        'q_fa_dot': q_fa_dot,

        # UIC at Busbar WTG1 LV
        'V_WTG1_LV_pu': abs(v_terminal),
        'angle_WTG1_LV_deg': np.degrees(np.angle(v_terminal)),
        'V_uic_internal_pu': abs(vi),
        'P_uic_bus_sys_pu': s_uic.real * uic_s_n / sys_s_n,
        'Q_uic_bus_sys_pu': s_uic.imag * uic_s_n / sys_s_n,
        'I_uic_pu': abs(i_a),
        'I_uic_angle_deg': np.degrees(np.angle(i_a)),

        # Remaining LEOGO synchronous generation
        'P_sync_generators_total_sys_pu': p_gen_total_sys_pu,
        'Q_sync_generators_total_sys_pu': q_gen_total_sys_pu,
    }


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-end', type=float, default=200.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--out', type=str, default=None,
                        help='Output CSV path (abs or relative to project root). '
                             'Defaults to results/WT1_LEOGO_tower_results.csv.')

    # Tower side-to-side (SS) mode controls
    parser.add_argument('--ss-off', action='store_true',
                        help='Disable the tower side-to-side mode '
                             '(then results match test_WT_LEOGO_sim.py).')
    parser.add_argument('--f-ss', type=float, default=0.234,
                        help='SS modal natural frequency [Hz].')
    parser.add_argument('--zeta-ss', type=float, default=0.05,
                        help='SS modal damping ratio [-].')
    parser.add_argument('--g-ss', type=float, default=0.2684,
                        help='SS torque->acceleration coupling gain [m/s^2 per pu] '
                             '(calibrated against the FMU sweep).')

    # Tower fore-aft (FA) mode controls (thrust-driven; calibrated vs FMU YawBrTAxp)
    parser.add_argument('--fa-off', action='store_true',
                        help='Disable the tower fore-aft mode.')
    parser.add_argument('--f-fa', type=float, default=0.235,
                        help='FA modal natural frequency [Hz].')
    parser.add_argument('--zeta-fa', type=float, default=0.0125,
                        help='FA modal (effective) damping ratio [-].')
    parser.add_argument('--g-fa', type=float, default=0.102,
                        help='FA thrust->acceleration coupling gain [m/s^2 per pu] '
                             '(calibrated against the FMU forced-resonance run).')

    # WindTurbineTower frequency-support droop. When --wt-droop is set, the
    # turbine reacts to the measured main-busbar frequency and raises its power
    # reference on a dip (using up to --headroom of de-loaded reserve). The
    # headroom de-loads the turbine even with droop off, so ON/OFF runs share
    # the same operating point and the SS difference isolates the droop action.
    parser.add_argument('--wt-droop', action='store_true',
                        help='Enable the WT frequency-support droop (headline '
                             'scenario). Off by default (MPT minus headroom).')
    parser.add_argument('--k-droop', type=float, default=0.75,
                        help='WT droop gain [pu power per Hz].')
    parser.add_argument('--headroom', type=float, default=0.10,
                        help='De-loaded reserve [pu on UIC base] kept for upward '
                             'frequency support (applied in both on/off runs).')

    # SS calibration/verification: sinusoidal modulation of the SS driving
    # torque, Te*(1 + amp*sin(2*pi*f*(t-start))). amp=0 (default) => no effect.
    # Reproduces the FMU sweep protocol on the reduced model.
    parser.add_argument('--ss-mod-amp', type=float, default=0.0,
                        help='Fractional SS torque-modulation amplitude '
                             '(0.10 = +/-10%%). 0 disables it.')
    parser.add_argument('--ss-mod-freq', type=float, default=0.234,
                        help='SS torque-modulation frequency [Hz].')
    parser.add_argument('--ss-mod-start', type=float, default=10.0,
                        help='Time [s] at which the SS modulation switches on.')

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

    # Under-frequency load shedding (GT-trip scenario): a second, permanent
    # load *reduction* at the same main grid bus, applied a short delay after
    # the load step. This reproduces the LEOGO under-frequency load-shedding
    # scheme (shed ~9 MW of water-injection pumps ~200 ms after a generator
    # trip; Svendsen et al. 2023, IET Energy Syst. Integr.).
    parser.add_argument('--load-shed-mw', type=float, default=0.0,
                        help='Permanent load reduction [MW] at the main grid '
                             'bus (under-frequency load shedding). 0 disables.')
    parser.add_argument('--load-shed-delay-s', type=float, default=0.2,
                        help='Delay [s] after --event-time before the load '
                             'shedding acts (LEOGO scheme: ~0.2 s).')

    # Optional three-phase fault at the WT connection point
    parser.add_argument('--fault', action='store_true')
    parser.add_argument('--fault-start', type=float, default=3.0)
    parser.add_argument('--fault-duration', type=float, default=0.05)
    parser.add_argument('--fault-admittance', type=float, default=1e6)

    # Sustained sinusoidal real-power load modulation at the main grid bus.
    # Unlike a load step (broadband, energy near DC), this is a persistent
    # excitation at a chosen frequency, so sweeping it probes the SS mode's
    # frequency selectivity through the genuine grid -> Pe -> Te -> SS path.
    parser.add_argument('--grid-mod-amp', type=float, default=0.0,
                        help='Amplitude [MW] of a sinusoidal load modulation at '
                             'the main grid bus. 0 disables it.')
    parser.add_argument('--grid-mod-freq', type=float, default=0.234,
                        help='Frequency [Hz] of the grid load modulation.')
    parser.add_argument('--grid-mod-start', type=float, default=10.0,
                        help='Time [s] at which the grid modulation switches on.')

    args = parser.parse_args()

    t_wall_start = time.perf_counter()

    # Build combined LEOGO + WT/UIC model
    model = build_model(
        ss_enable=0 if args.ss_off else 1,
        f_ss_hz=args.f_ss,
        zeta_ss=args.zeta_ss,
        g_ss=args.g_ss,
        fa_enable=0 if args.fa_off else 1,
        f_fa_hz=args.f_fa,
        zeta_fa=args.zeta_fa,
        g_fa=args.g_fa,
        droop_enable=1 if args.wt_droop else 0,
        k_droop_pu_per_hz=args.k_droop,
        headroom_pu=args.headroom,
    )

    ps = dps.PowerSystemModel(
        model=model,
        user_mdl_lib=ext_lib,
    )

    wt_model = ps.windturbine_tower['WindTurbineTower']
    uic_model = ps.vsc['UIC_sig']
    gen_model = ps.gen['GEN']

    # Initialise converter active-power reference from turbine MPT curve
    wind_speed_initial = wt_model.wind_speed_init()
    p_ref_initial = wt_model.P_ref_from_wind(
        wind_speed_initial,
        uic_model.par['S_n'],
    )

    uic_model.par['p_ref'][:] = p_ref_initial
    uic_model.par['q_ref'][:] = 0.0

    # Power flow and dynamic initialisation
    ps.power_flow()
    ps.init_dyn_sim()

    x0 = ps.x0.copy()

    print('\nInitialised LEOGO + WT1 model (WindTurbineTower)')
    print(f"WT UIC name: {uic_model.par['name'][0]}")
    print(f"WT electrical bus: {uic_model.par['bus'][0]}")
    print(f"Initial wind speed: {scalar(wind_speed_initial):.3f} m/s")
    print(f"Initial UIC P reference: {scalar(p_ref_initial):.5f} pu on UIC base")
    print(
        f"Tower SS mode: {'OFF' if args.ss_off else 'ON'} "
        f"(f_ss={args.f_ss:.3f} Hz, zeta={args.zeta_ss:.3f}, g_ss={args.g_ss:.3f})"
    )
    print(
        f"Tower FA mode: {'OFF' if args.fa_off else 'ON'} "
        f"(f_fa={args.f_fa:.3f} Hz, zeta={args.zeta_fa:.4f}, g_fa={args.g_fa:.3f})"
    )
    print(
        f"WT frequency support: {'ON' if args.wt_droop else 'OFF'} "
        f"(K_droop={args.k_droop:.3f} pu/Hz, headroom={args.headroom:.3f} pu)"
    )

    # Optional SS torque modulation for calibration/verification sweeps.
    if args.ss_mod_amp != 0.0:
        amp = float(args.ss_mod_amp)
        f_mod = float(args.ss_mod_freq)
        t_mod = float(args.ss_mod_start)
        w_mod = 2.0 * np.pi * f_mod

        def ss_modulation(t):
            return amp * np.sin(w_mod * (t - t_mod)) if t >= t_mod else 0.0

        wt_model.set_ss_test_modulation(ss_modulation)
        print(
            f"SS torque modulation: +/-{amp*100:.0f}% at {f_mod:.3f} Hz "
            f"from t={t_mod:.1f} s"
        )

    if args.grid_mod_amp != 0.0:
        print(
            f"Grid load modulation: +/-{args.grid_mod_amp:.2f} MW at "
            f"{args.grid_mod_freq:.3f} Hz from t={args.grid_mod_start:.1f} s"
        )

    if args.load_shed_mw != 0.0:
        print(
            f"Load shedding: -{args.load_shed_mw:.2f} MW at "
            f"t={args.event_time + args.load_shed_delay_s:.2f} s (GT-trip scheme)"
        )

    # The reduced-network index of Busbar WTG1 LV
    fault_bus_idx = uic_model.bus_idx_red['terminal'][0]

    # Reduced-network index of the main grid (gas-turbine) bus, where the
    # smooth load step is applied. This is the network-side frequency event.
    s_base_mva = float(model['base_mva'])
    load_bus_idx = gen_model.bus_idx_red['terminal'][0]
    y_load_step = (
        args.load_step_mw / s_base_mva - 1j * args.load_step_mvar / s_base_mva
    )

    # Sinusoidal grid-bus load modulation (real power only, base admittance).
    y_grid_mod = args.grid_mod_amp / s_base_mva

    # Under-frequency load shedding: a permanent real-power load *reduction* at
    # the main grid bus, switched in a short delay after the load step (GT-trip
    # scenario, LEOGO scheme). Enters set_load_step with a minus sign.
    y_load_shed = args.load_shed_mw / s_base_mva
    load_shed_ramp_s = 0.02

    def grid_mod_scale(t):
        """Signed sinusoid in [-1, 1] for the grid load modulation (0 if off)."""
        if args.grid_mod_amp == 0.0 or t < args.grid_mod_start:
            return 0.0
        return float(
            np.sin(2.0 * np.pi * args.grid_mod_freq * (t - args.grid_mod_start))
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
        ramp is used and the shed load stays off for the rest of the run.
        """
        if args.load_shed_mw == 0.0:
            return 0.0
        t_shed = args.event_time + args.load_shed_delay_s
        if t < t_shed:
            return 0.0
        if load_shed_ramp_s > 0.0 and t < t_shed + load_shed_ramp_s:
            return smoothstep((t - t_shed) / load_shed_ramp_s)
        return 1.0

    def set_load_step(t):
        # Combined grid-bus disturbance: smooth load step + sinusoidal modulation
        # + permanent under-frequency load shedding (all act on the same main
        # grid bus, so they are summed; the shed load enters with a minus sign).
        ps.y_bus_red_mod[(load_bus_idx, load_bus_idx)] = (
            load_event_scale(t) * y_load_step
            + grid_mod_scale(t) * y_grid_mod
            - load_shed_scale(t) * y_load_shed
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

    # Main-busbar frequency fed to the WT frequency-support droop. The GT speed
    # is a state, so it is available from x before the algebraic solve; the WT
    # droop reads self._grid_frequency_hz when it forms P_ref during that solve.
    f_n_grid = float(np.asarray(gen_model.sys_par['f_n']).ravel()[0])

    def feed_grid_frequency(x):
        gen_speed_pu = np.asarray(gen_model.speed(x, None), dtype=float).ravel()
        f_grid = f_n_grid * (1.0 + float(np.mean(gen_speed_pu)))
        wt_model.set_grid_frequency_hz(f_grid)

    def f_ode(t, x):
        # The WT wind lookup uses this time internally.
        wt_model._sim_time = t

        # Set disturbances before the algebraic solve during each RK4 stage.
        set_load_step(t)
        set_fault(t)

        # Feed the measured grid frequency to the WT droop before P_ref forms.
        feed_grid_frequency(x)

        v = ps.solve_algebraic(t, x)
        return ps.state_derivatives(t, x, v)

    solver = dps_sol.SimpleRK4(
        f_ode,
        0.0,
        x0,
        args.t_end,
        max_step=args.dt,
    )

    # Store t = 0 point
    wt_model._sim_time = 0.0
    set_load_step(0.0)
    set_fault(0.0)
    v0 = ps.solve_algebraic(0.0, x0)

    rows = [
        collect_results(
            ps, wt_model, uic_model, gen_model,
            t=0.0, x=x0, v=v0,
        )
    ]
    rows[0]['load_step_scale'] = load_event_scale(0.0)
    rows[0]['grid_mod_mw'] = args.grid_mod_amp * grid_mod_scale(0.0)
    rows[0]['load_shed_mw'] = args.load_shed_mw * load_shed_scale(0.0)

    while solver.t < args.t_end:
        solver.step()

        t = solver.t
        x = solver.x

        wt_model._sim_time = t
        set_load_step(t)
        set_fault(t)
        v = ps.solve_algebraic(t, x)

        row = collect_results(
            ps, wt_model, uic_model, gen_model,
            t=t, x=x, v=v,
        )
        row['load_step_scale'] = load_event_scale(t)
        row['grid_mod_mw'] = args.grid_mod_amp * grid_mod_scale(t)
        row['load_shed_mw'] = args.load_shed_mw * load_shed_scale(t)
        rows.append(row)

        progress = min(100, int(100 * t / args.t_end))
        print(f'\rSimulation progress: {progress:3d}%', end='', flush=True)

    print('\rSimulation progress: 100%')

    results = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / 'casestudies' / 'dyn_sim' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.out is not None:
        out_path = Path(args.out)
        output_file = out_path if out_path.is_absolute() else PROJECT_ROOT / out_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = output_dir / 'WT1_LEOGO_tower_results.csv'

    results.to_csv(output_file, index=False)

    print(f'Results written to: {output_file}')
    print(f'Simulation wall time: {time.perf_counter() - t_wall_start:.2f} s')

    if args.show:
        fig, ax = plt.subplots()
        ax.plot(results['t'], results['ss_accel_mps2'])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Tower side-to-side acceleration [m/s$^2$]')
        ax.grid(True)

        fig, ax = plt.subplots()
        ax.plot(results['t'], results['P_uic_bus_sys_pu'], label='UIC electrical power')
        ax.plot(results['t'], results['P_aero_sys_pu'], label='Aerodynamic power')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Power [pu on system base]')
        ax.grid(True)
        ax.legend()

        plt.show()


if __name__ == '__main__':
    main()
