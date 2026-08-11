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

def build_model():
    """
    Start from the LEOGO electrical network and replace its original,
    simplified WT1 grid-side converter with the WT + UIC_sig model.
    """
    model = deepcopy(load_leogo())

    wt_bus = 'Busbar WTG1 LV'

    # -------------------------------------------------------------
    # 1. Remove the original LEOGO converter at WTG1.
    #
    # This removes:
    # model['vsc']['GridSideConverter_PV']
    #
    # Its former component name was WT1_LEOGO. We reuse that name
    # for the UIC_sig so the WT model can connect to it.
    # -------------------------------------------------------------
    old_converter = model['vsc'].pop('GridSideConverter_PV', None)

    if old_converter is None:
        raise KeyError(
            "Could not find model['vsc']['GridSideConverter_PV']. "
            "Check LEOGO_ps.py before running."
        )

    # -------------------------------------------------------------
    # 2. Add the UIC model at the same physical electrical bus.
    #
    # The name WT1_LEOGO is now the UIC component name.
    # The actual electrical bus remains Busbar WTG1 LV.
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
            1,
            0.01,
        ],
    ]

    # -------------------------------------------------------------
    # 3. Add the aerodynamic WT model.
    #
    # WT1 connects to the UIC component called WT1_LEOGO.
    # This is not the bus name.
    # -------------------------------------------------------------
    model['windturbine'] = {
        'WindTurbine': [
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
                'f_nom_hz','droop_enable','K_droop_pu_per_hz','headroom_pu'
            ],
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
                1, 0.75, 0.05,
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

    LEOGO has three synchronous generators, so generator power is
    summed instead of assuming a single infinite-bus machine.
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

    return {
        't': t,

        # Wind turbine
        'wind_speed_mps': scalar(wt_model.wind_speed(x, v)),
        'omega_m_pu': omega_m_pu,
        'omega_e_pu': omega_e_pu,
        'pitch_deg': np.degrees(pitch_rad),
        'P_aero_sys_pu': scalar(wt_model.P_aero(x, v)) * wt_s_n / sys_s_n,
        'P_e_sys_pu': scalar(wt_model.P_e(x, v)) * uic_s_n / sys_s_n,
        'P_ref_sys_pu': scalar(wt_model.P_ref(x, v)) * uic_s_n / sys_s_n,

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

    # Smooth load step applied as a temporary shunt at the main grid bus.
    # Matches test_WT_LEOGO_FMU_sim.py so the two runs are directly comparable.
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

    # Optional three-phase fault at the WT connection point
    parser.add_argument('--fault', action='store_true')
    parser.add_argument('--fault-start', type=float, default=3.0)
    parser.add_argument('--fault-duration', type=float, default=0.05)
    parser.add_argument('--fault-admittance', type=float, default=1e6)

    args = parser.parse_args()

    t_wall_start = time.perf_counter()

    # Build combined LEOGO + WT/UIC model
    model = build_model()

    ps = dps.PowerSystemModel(
        model=model,
        user_mdl_lib=ext_lib,
    )

    wt_model = ps.windturbine['WindTurbine']
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

    print('\nInitialised LEOGO + WT1 model')
    print(f"WT UIC name: {uic_model.par['name'][0]}")
    print(f"WT electrical bus: {uic_model.par['bus'][0]}")
    print(f"Initial wind speed: {scalar(wind_speed_initial):.3f} m/s")
    print(f"Initial UIC P reference: {scalar(p_ref_initial):.5f} pu on UIC base")

    # The reduced-network index of Busbar WTG1 LV
    fault_bus_idx = uic_model.bus_idx_red['terminal'][0]

    # Reduced-network index of the main grid (gas-turbine) bus, where the
    # smooth load step is applied. This is the network-side frequency event.
    s_base_mva = float(model['base_mva'])
    load_bus_idx = gen_model.bus_idx_red['terminal'][0]
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

    def set_load_step(t):
        ps.y_bus_red_mod[(load_bus_idx, load_bus_idx)] = (
            load_event_scale(t) * y_load_step
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

    def f_ode(t, x):
        # The WT wind lookup uses this time internally.
        wt_model._sim_time = t

        # Set disturbances before the algebraic solve during each RK4 stage.
        set_load_step(t)
        set_fault(t)

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
        rows.append(row)

        progress = min(100, int(100 * t / args.t_end))
        print(f'\rSimulation progress: {progress:3d}%', end='', flush=True)

    print('\rSimulation progress: 100%')

    results = pd.DataFrame(rows)

    output_dir = PROJECT_ROOT / 'casestudies' / 'dyn_sim' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'WT1_LEOGO_results.csv'
    results.to_csv(output_file, index=False)

    print(f'Results written to: {output_file}')
    print(f'Simulation wall time: {time.perf_counter() - t_wall_start:.2f} s')

    if args.show:
        fig, ax = plt.subplots()
        ax.plot(results['t'], results['V_WTG1_LV_pu'])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('|V| at Busbar WTG1 LV [pu]')
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