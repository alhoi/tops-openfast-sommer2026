import sys
import os
import argparse
# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
from pathlib import Path
project_root = Path(__file__).parents[2]
if project_root not in sys.path:
     sys.path.insert(0, project_root)

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from casestudies.dyn_sim.plotting.log_paths import WT_CSV, ensure_log_dir, wt_thesis_plots_dir
from casestudies.dyn_sim.plotting.thesis_plot_style import save_baseline_thesis_plots
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)

if __name__ == '__main__':
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument('--show', action='store_true', help='Open plot windows (default: save PNGs only).')
    _ap.add_argument('--no-thesis-plots', action='store_true')
    _cli, _ = _ap.parse_known_args()

    t_start_wall = time.perf_counter()

    # region Model loading and initialisation
    import casestudies.ps_data.test_WT as model_data
    model = model_data.load()
    import tops_openfast.dyn_models as ext_lib
    
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)  # Load into a PowerSystemModel object

    # Set UIC p_ref from WT MPT - use WT's wind_speed_init() so it always matches
    wt_model = ps.windturbine['WindTurbine']
    wind_speed = wt_model.wind_speed_init()
    uic_model = ps.vsc['UIC_sig']
    #uic_model = ps.vsc['UIC_sig_pq']
    P_ref = wt_model.P_ref_from_wind(wind_speed, uic_model.par['S_n'])
    uic_model.par['p_ref'][:] = P_ref
    uic_model.par['q_ref'][:] = 0.0 # will be overridden for pv bus, will be affected by xf loss (visible in pq run)

    ps.power_flow()  # Power flow calculation

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states
    v0 = ps.v0.copy()

    gen_model = ps.gen['GEN']  # Infinite bus generator
    
    wt_name = wt_model.par['name'][0]
    uic_name = uic_model.par['name'][0]
    gen_name = gen_model.par['name'][0]

    t = 0
    result_dict = defaultdict(list)
    t_end = 10. # Simulation time
    # t_end = 40.0  # Simulation time

    # Solver
    # Test explicit RK4 on the differential states. Since this is a DAE, solve algebraics explicitly inside f_ode.
    dt = 0.01
    def f_ode(t_, x_):
        return ps.state_derivatives(t_, x_, ps.solve_algebraic(t_, x_))
    
    sol = dps_sol.SimpleRK4(f_ode, 0.0, x0, t_end, max_step=dt)
    # endregion

    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = np.angle(ps.v_0)  # In radians
    print(f'Voltages (pu): {v_bus_mag}')
    print(f'Voltage angles: {v_bus_angle} \n')
    print(f'state description: \n {ps.state_desc} \n')
    print(f'Initial values on all state variables (WT and UIC) : \n {x0} \n')
    
    # region Runtime variables
    # Additional plot variables
    P_aero_stored = []
    P_e_stored = []
    P_ref_stored = []
    P_ref_instant_stored = []  # instantaneous MPT ref (before lag), for comparison
    v_bus = []
    vi_mag_hist = []
    omega_m_hist = []
    omega_e_hist = []
    T_mpt_wt_pu_hist = []
    pitch_angle_hist = []
    wind_speed_hist = []
    i_a_mag_hist = []
    i_a_angle_hist = []
    P_gen_stored = [] 
    Q_gen_stored = [] 
    # Bus-side UIC power (actual and reference)
    P_uic_bus_actual = []
    Q_uic_bus_actual = []
    P_uic_bus_ref = []
    Q_uic_bus_ref = []

    # endregion

    # Store initial point (t0=0, x0, v0) so plots include first time step.
    # Use the same algebraic solution as the DAE solver (solve_algebraic(0,x0)), not power-flow voltage.
    # Otherwise the first point uses PF v while the solver uses Y*v=i_inj(x0) v, causing a visible jump.
    wt_model._sim_time = 0  # t=0 for wind file lookup at init
    v0 = ps.solve_algebraic(0, x0)
    result_dict['Global', 't'].append(0)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x0)]
    sys_s_n = wt_model.sys_par['s_n']
    uic_s_n = uic_model.par['S_n'][0]
    wt_s_n = wt_model.par['S_n'][0]
    gen_s_n = gen_model.par['S_n'][0]
    omega_rated_rad_s = float(np.asarray(wt_model.par['omega_m_rated']).ravel()[0])

    def _append_mpt_torque(wt_states):
        om_e_pu = float(np.asarray(wt_states['omega_e']).ravel()[0])
        T_mpt_wt_pu_hist.append(
            float(wt_model._mpt_torque_mech_pu(om_e_pu * omega_rated_rad_s))
        )

    v_t_uic = uic_model.v_t(x0, v0)[0]
    v_bus.append(np.abs(v_t_uic))
    P_aero_local = wt_model.P_aero(x0, v0)[0]
    P_e_uic = wt_model.P_e(x0, v0)[0]
    P_ref_uic = wt_model.P_ref(x0, v0)[0]
    P_aero_stored.append(P_aero_local * wt_s_n / sys_s_n)
    P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
    P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)
    # No lag in P_ref anymore; treat "instant" as the same as P_ref (UIC pu) for comparison.
    P_ref_instant_stored.append(float(np.atleast_1d(wt_model.P_ref(x0, v0)).flat[0]) * uic_s_n / sys_s_n)
    P_gen_local = gen_model.p_e(x0, v0)[0]
    Q_gen_local = gen_model.q_e(x0, v0)[0]
    P_gen_stored.append(P_gen_local * gen_s_n / sys_s_n)
    Q_gen_stored.append(Q_gen_local * gen_s_n / sys_s_n)

    # UIC bus-side actual and reference at t0
    X = uic_model.local_view(x0)
    vi = X['vi_x'][0] + 1j*X['vi_y'][0]
    vi_mag_hist.append(float(np.abs(vi)))
    i_a = uic_model.i_a(x0, v0)[0]
    s_bus_actual = uic_model.s_e(x0, v0)[0]  # bus-side S
    # Internal reference S at vi
    s_ref_internal = uic_model.p_ref(x0, v0)[0] + 1j * uic_model.q_ref(x0, v0)[0]
    # Transform internal reference to bus-side: S_ext = S_int - j*xf*|I_a|^2
    xf = uic_model.par['xf'][0]
    s_bus_ref = s_ref_internal - 1j * xf * (np.abs(i_a) ** 2)

    P_uic_bus_actual.append(s_bus_actual.real * uic_s_n / sys_s_n)
    Q_uic_bus_actual.append(s_bus_actual.imag * uic_s_n / sys_s_n)
    P_uic_bus_ref.append(s_bus_ref.real * uic_s_n / sys_s_n)
    Q_uic_bus_ref.append(s_bus_ref.imag * uic_s_n / sys_s_n)
    wt_states = wt_model.local_view(x0)
    omega_m_hist.append(wt_states['omega_m'][0])
    omega_e_hist.append(wt_states['omega_e'][0])
    _append_mpt_torque(wt_states)
    pitch_angle_val = wt_states['pitch_angle'][0] if 'pitch_angle' in wt_states.dtype.names else 0.0
    pitch_angle_hist.append(float(pitch_angle_val * 180 / np.pi))
    wind_speed_hist.append(wt_model.wind_speed(x0, v0))
    i_a_mag_hist.append(np.abs(i_a))
    i_a_angle_hist.append(np.angle(i_a) * 180 / np.pi)

    # Simulation loop starts here!
    # Short circuit parameters (modify reduced Ybus diagonal at the chosen bus)
    t_sc = 120.0
    t_sc_dur = 0.05
    y_sc = 1e6
    sc_flag = False
    while t < t_end:
        # Progress indicator (single-line percentage)
        sys.stdout.write("\r%d%%" % int((sol.t / t_end) * 100))
        sys.stdout.flush()
        wt_model._sim_time = sol.t  # set before step so WT wind lookup uses correct time
        result = sol.step()
        x = sol.x
        t = sol.t
        v = ps.solve_algebraic(t, x)
        wt_model._sim_time = t  # update for storage/plot

        sc_bus_idx = ps.vsc['UIC_sig'].bus_idx_red['terminal'][0]
        #sc_bus_idx = ps.vsc['UIC_sig_pq'].bus_idx_red['terminal'][0]

        # Short circuit (apply at UIC terminal bus)
        if t_sc <= t <= (t_sc + t_sc_dur) and sc_flag:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = y_sc
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables

        v_t_uic = uic_model.v_t(x, v)[0]
        v_bus.append(np.abs(v_t_uic))
        vi_mag_hist.append(float(np.abs(vi)))
        P_aero_local = wt_model.P_aero(x, v)[0] 
        P_e_uic = wt_model.P_e(x, v)[0]  
        P_ref_uic = wt_model.P_ref(x, v)[0]  
        sys_s_n = wt_model.sys_par['s_n']
        uic_s_n = uic_model.par['S_n'][0]
        wt_s_n = wt_model.par['S_n'][0]
        gen_s_n = gen_model.par['S_n'][0]
        P_aero_stored.append(P_aero_local * wt_s_n / sys_s_n)  # WT local → system
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)  
        P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)  
        P_ref_instant_stored.append(float(np.atleast_1d(wt_model.P_ref(x, v)).flat[0]) * uic_s_n / sys_s_n)
        P_gen_local = gen_model.p_e(x, v)[0]  
        Q_gen_local = gen_model.q_e(x, v)[0]  
        P_gen_stored.append(P_gen_local * gen_s_n / sys_s_n)  
        Q_gen_stored.append(Q_gen_local * gen_s_n / sys_s_n)  

        # UIC bus-side actual and reference
        X = uic_model.local_view(x)
        vi = X['vi_x'][0] + 1j*X['vi_y'][0]  # Internal voltage
        i_a = uic_model.i_a(x, v)[0]  # Current through xf (from terminal to internal)
        s_bus_actual = uic_model.s_e(x, v)[0]
        s_ref_internal = uic_model.p_ref(x, v)[0] + 1j * uic_model.q_ref(x, v)[0]
        xf = uic_model.par['xf'][0]
        s_bus_ref = s_ref_internal - 1j * xf * (np.abs(i_a) ** 2)

        P_uic_bus_actual.append(s_bus_actual.real * uic_s_n / sys_s_n)
        Q_uic_bus_actual.append(s_bus_actual.imag * uic_s_n / sys_s_n)
        P_uic_bus_ref.append(s_bus_ref.real * uic_s_n / sys_s_n)
        Q_uic_bus_ref.append(s_bus_ref.imag * uic_s_n / sys_s_n)
        wt_states = wt_model.local_view(x)
        omega_m_hist.append(wt_states['omega_m'][0])
        omega_e_hist.append(wt_states['omega_e'][0])
        _append_mpt_torque(wt_states)
        # Pitch angle is stored as state variable
        pitch_angle_val = wt_states['pitch_angle'][0] if 'pitch_angle' in wt_states.dtype.names else 0.0
        pitch_angle_hist.append(float(pitch_angle_val * 180 / np.pi))  # Convert to degrees and ensure scalar
        # Wind speed
        wind_speed_hist.append(wt_model.wind_speed(x, v))
        # UIC armature current
        i_a = uic_model.i_a(x, v)[0]
        i_a_mag_hist.append(np.abs(i_a))
        i_a_angle_hist.append(np.angle(i_a) * 180 / np.pi)  # Convert to degrees
        # endregion

    sys.stdout.write("\r100%\n")
    sys.stdout.flush()

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

    # Export CSV so results can be compared to the FMU drivetrain simulation.
    t_stored = result[('Global', 't')]
    # Speed base used for omega_*_pu in this simulation (WindTurbine internally uses omega_m_rated as base).
    omega_base_rad_s = float(np.asarray(wt_model.par['omega_m_rated']).ravel()[0])
    omega_base_rpm = omega_base_rad_s * 60.0 / (2.0 * np.pi) if np.isfinite(omega_base_rad_s) else np.nan
    out_df = pd.DataFrame(
        {
            't': t_stored.to_numpy(dtype=float),
            # Wind turbine (system-base pu where noted)
            'omega_base_rpm': float(omega_base_rpm),
            'omega_m_pu': np.asarray(omega_m_hist, dtype=float),
            'omega_e_pu': np.asarray(omega_e_hist, dtype=float),
            'T_mpt_wt_pu': np.asarray(T_mpt_wt_pu_hist, dtype=float),
            'pitch_deg': np.asarray(pitch_angle_hist, dtype=float),
            'wind_speed_mps': np.asarray(wind_speed_hist, dtype=float),
            'P_aero_sys_pu': np.asarray(P_aero_stored, dtype=float),
            'P_e_sys_pu': np.asarray(P_e_stored, dtype=float),
            'P_ref_sys_pu': np.asarray(P_ref_stored, dtype=float),
            'P_ref_instant_sys_pu': np.asarray(P_ref_instant_stored, dtype=float),
            # UIC voltages (terminal |V_t| and internal |v_i|, system V base pu)
            'v_bus_pu': np.asarray(v_bus, dtype=float),
            'vi_mag_pu': np.asarray(vi_mag_hist, dtype=float),
            # UIC bus-side power (sys base pu)
            'P_uic_bus_actual_sys_pu': np.asarray(P_uic_bus_actual, dtype=float),
            'Q_uic_bus_actual_sys_pu': np.asarray(Q_uic_bus_actual, dtype=float),
            'P_uic_bus_ref_sys_pu': np.asarray(P_uic_bus_ref, dtype=float),
            'Q_uic_bus_ref_sys_pu': np.asarray(Q_uic_bus_ref, dtype=float),
            # Infinite bus power (sys base pu)
            'P_inf_sys_pu': np.asarray(P_gen_stored, dtype=float),
            'Q_inf_sys_pu': np.asarray(Q_gen_stored, dtype=float),
            # UIC armature current (magnitude pu on UIC base, angle deg)
            'i_a_mag_pu_uic': np.asarray(i_a_mag_hist, dtype=float),
            'i_a_angle_deg': np.asarray(i_a_angle_hist, dtype=float),
        }
    )

    ensure_log_dir(WT_CSV)
    out_df.to_csv(WT_CSV, index=False)
    print(f"\nResults saved to {WT_CSV} ({len(out_df.columns)} columns)")

    if not _cli.no_thesis_plots:
        thesis_dir = wt_thesis_plots_dir()
        paths = save_baseline_thesis_plots(
            str(thesis_dir), out_df['t'].to_numpy(dtype=float), out_df, show=_cli.show,
        )
        print(f"Thesis figures saved to {thesis_dir} ({len(paths)} files)")

    print(f"\nSimulation took {time.perf_counter() - t_start_wall:.2f} seconds.")
    if _cli.show and not _cli.no_thesis_plots:
        plt.show(block=True)
    # endregion
