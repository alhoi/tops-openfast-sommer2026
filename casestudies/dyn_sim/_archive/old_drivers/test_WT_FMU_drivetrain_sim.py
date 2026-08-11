import sys
import os
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import tops.dynamic as dps
import tops.solvers as dps_sol

from casestudies.dyn_sim.plotting.log_paths import (
    FMU_DRIVETRAIN_CSV,
    ensure_log_dir,
    fmu_drivetrain_thesis_plots_dir,
)
from casestudies.dyn_sim.plotting.thesis_plot_style import save_coupled_thesis_plots
from pathlib import Path

# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(script_dir))
# if project_root not in sys.path:
    # sys.path.insert(0, project_root)
project_root = Path(__file__).parents[2]
# Like DynaWind: cwd = project root so extract(unzipdir='openfast_fmu') and wd path work
os.chdir(project_root)

os.chdir(project_root)

def _safe_legend(ax, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(*args, **kwargs)

if __name__ == '__main__':
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument('--show', action='store_true', help='Open plot windows (default: save PNGs only).')
    _ap.add_argument('--no-thesis-plots', action='store_true')
    _cli, _ = _ap.parse_known_args()

    t_start_wall = time.perf_counter()
    # Model loading and initialisation
    print("Loading model data...", flush=True)
    import casestudies.ps_data.test_WT_FMU_drivetrain_ as model_data
    model = model_data.load()
    print("Building PowerSystemModel...", flush=True)
    import tops_openfast.dyn_models as ext_lib
    ps = dps.PowerSystemModel(
        model=model, user_mdl_lib=ext_lib
    )  # Load into a PowerSystemModel object

    # UIC p_ref for power flow (FMU provides it during dynamics via connection)
    uic_model = ps.vsc['UIC_sig']
    gen_model = ps.gen['GEN']
    uic_model.par['p_ref'][:] = 0.75 #0.6471503597375907/2 
    uic_model.par['q_ref'][:] = 0.

    t0 = time.perf_counter()
    print("Running power flow...", flush=True)
    ps.power_flow()
    print(f"Power flow done in {time.perf_counter()-t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    print("Initializing dynamic simulation (init_dyn_sim)...", flush=True)
    ps.init_dyn_sim()
    print(f"init_dyn_sim done in {time.perf_counter()-t0:.2f}s", flush=True)
    x0 = ps.x0.copy()
    v0 = ps.v0.copy()

    fmu_models = [mdl for mdl in ps.dyn_mdls if hasattr(mdl, 'step_fmu')]
    t = 0.0
    result_dict = defaultdict(list)
    # Allow quick A/B interface tests without changing the file.
    # Example: set FMU_T_END=20 to run 20 seconds.xx
    t_end = float(os.getenv('FMU_T_END', '240.'))
    dt = 0.01
    # Use dt=0.01 to match OpenFAST FMU (canHandleVariableCommunicationStepSize=false)
    f_ode = lambda t_, x_: ps.state_derivatives(t_, x_, ps.solve_algebraic(t_, x_))
    sol = dps_sol.SimpleRK4(f_ode, 0.0, x0, t_end, max_step=dt)
    # Keep explicit current state variables (used for FMU-first co-simulation ordering).
    x = x0
    v = v0

    # Runtime storage (all FMU outputs from modelDescription.xml)
    P_e_stored = []
    P_ref_stored = []
    P_e_uic_pu_stored = []
    P_ref_uic_pu_stored = []
    # UIC bus-side complex power (same definitions as test_WT_sim.py / wt_model.csv)
    P_uic_bus_actual_stored = []
    Q_uic_bus_actual_stored = []
    P_uic_bus_ref_stored = []
    Q_uic_bus_ref_stored = []
    v_bus = []
    vi_mag_hist = []
    i_a_mag_hist = []
    i_a_angle_hist = []
    P_inf_stored = []
    Q_inf_stored = []
    fmu_outputs_stored = []
    # Also store commanded electrical torque sent to FMU (from FMUtoUICdrivetrain)
    Te_cmd_pu_stored = []
    Te_cmd_kNm_stored = []
    # Store the torque value written to FMU input GenSpdOrTrq (debugging, kN·m)
    GenSpdOrTrq_set_kNm_stored = []
    # Store the effective omega_m measurement used by the wrapper (pu)
    omega_m_pu_meas_stored = []
    # Store scaled power inputs written to the FMU (debugging)
    GenPwr_set_kW_stored = []
    ElecPwrCom_set_kW_stored = []

    # Initial point
    t0 = time.perf_counter()
    print("Solving algebraic equations at t=0...", flush=True)
    v0 = ps.solve_algebraic(0.0, x0)
    print(f"solve_algebraic(t=0) done in {time.perf_counter()-t0:.2f}s", flush=True)
    result_dict['Global', 't'].append(0.0)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x0)]
    sys_s_n = ps.sys_data['s_n']
    uic_s_n = uic_model.par['S_n'][0]
    gen_s_n = gen_model.par['S_n'][0]
    # Short circuit parameters (modify reduced Ybus diagonal at the chosen bus)
    sc_bus_idx = ps.vsc['UIC_sig'].bus_idx_red['terminal'][0]
    run_sc = False
    t_sc = 60.
    t_sc_dur = 0.05
    y_sc = 1e6

    P_e_uic = uic_model.p_e(x0, v0)[0]
    P_ref_uic = uic_model.p_ref(x0, v0)[0]
    P_e_uic_pu_stored.append(P_e_uic)
    P_ref_uic_pu_stored.append(P_ref_uic)
    P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
    P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

    fmu_mdl = fmu_models[0] if fmu_models else None
    # Do not assign/plot FMU outputs at t=0 before we've latched a valid post-step FMU sample.
    # Seed an "all-NaN" output row so exported columns exist and plots clearly show missing values.
    if fmu_mdl is not None and hasattr(fmu_mdl, 'FMU_OUTPUT_NAMES'):
        # Only seed outputs the FMU actually exposes (modelDescription.xml / self.vrs).
        d0 = {
            name: np.nan
            for name in getattr(fmu_mdl, 'FMU_OUTPUT_NAMES', [])
            if name in getattr(fmu_mdl, 'vrs', {})
        }
        d0['Time'] = float(t)
        d0['Time_fmu'] = np.nan
        fmu_outputs_stored.append(d0)
    elif fmu_mdl is not None:
        fmu_outputs_stored.append({'Time': float(t), 'Time_fmu': np.nan})
    if fmu_mdl is not None and hasattr(fmu_mdl, '_Te_pu_cmd'):
        te_pu = (
            float(np.asarray(fmu_mdl._Te_pu_cmd).ravel()[0])
            if fmu_mdl._Te_pu_cmd is not None and np.asarray(fmu_mdl._Te_pu_cmd).size > 0
            else np.nan
        )
        Te_cmd_pu_stored.append(te_pu)
        if hasattr(fmu_mdl, '_T_base_Nm') and np.isfinite(te_pu):
            T_base_Nm = float(np.asarray(fmu_mdl._T_base_Nm).ravel()[0])
            Te_cmd_kNm_stored.append(te_pu * T_base_Nm / 1e3)
        else:
            Te_cmd_kNm_stored.append(np.nan)
    else:
        Te_cmd_pu_stored.append(np.nan)
        Te_cmd_kNm_stored.append(np.nan)
    if fmu_mdl is not None and hasattr(fmu_mdl, '_gen_spdortrq_kNm_set'):
        GenSpdOrTrq_set_kNm_stored.append(
            float(fmu_mdl._gen_spdortrq_kNm_set) if fmu_mdl._gen_spdortrq_kNm_set is not None else np.nan
        )
    else:
        GenSpdOrTrq_set_kNm_stored.append(np.nan)

    if fmu_mdl is not None and hasattr(fmu_mdl, '_omega_m_pu_meas'):
        omega_m_pu_meas_stored.append(
            float(fmu_mdl._omega_m_pu_meas) if fmu_mdl._omega_m_pu_meas is not None else np.nan
        )
    else:
        omega_m_pu_meas_stored.append(np.nan)

    if fmu_mdl is not None and hasattr(fmu_mdl, '_genpwr_kW_set'):
        GenPwr_set_kW_stored.append(
            float(fmu_mdl._genpwr_kW_set) if fmu_mdl._genpwr_kW_set is not None else np.nan
        )
    else:
        GenPwr_set_kW_stored.append(np.nan)

    if fmu_mdl is not None and hasattr(fmu_mdl, '_elec_pwr_com_kW_last'):
        ElecPwrCom_set_kW_stored.append(
            float(fmu_mdl._elec_pwr_com_kW_last) if fmu_mdl._elec_pwr_com_kW_last is not None else np.nan
        )
    else:
        ElecPwrCom_set_kW_stored.append(np.nan)


    X0 = uic_model.local_view(x0)
    vi0 = X0['vi_x'][0] + 1j * X0['vi_y'][0]
    i_a0 = uic_model.i_a(x0, v0)[0]
    s_bus_actual0 = uic_model.s_e(x0, v0)[0]
    s_ref_internal0 = uic_model.p_ref(x0, v0)[0] + 1j * uic_model.q_ref(x0, v0)[0]
    xf0 = uic_model.par['xf'][0]
    s_bus_ref0 = s_ref_internal0 - 1j * xf0 * (np.abs(i_a0) ** 2)
    P_uic_bus_actual_stored.append(s_bus_actual0.real * uic_s_n / sys_s_n)
    Q_uic_bus_actual_stored.append(s_bus_actual0.imag * uic_s_n / sys_s_n)
    P_uic_bus_ref_stored.append(s_bus_ref0.real * uic_s_n / sys_s_n)
    Q_uic_bus_ref_stored.append(s_bus_ref0.imag * uic_s_n / sys_s_n)

    v_t_uic = uic_model.v_t(x0, v0)[0]
    v_bus.append(np.abs(v_t_uic))
    vi_mag_hist.append(float(np.abs(vi0)))
    i_a_mag_hist.append(float(np.abs(i_a0)))
    i_a_angle_hist.append(float(np.angle(i_a0) * 180 / np.pi))
    P_gen0 = gen_model.p_e(x0, v0)[0]
    Q_gen0 = gen_model.q_e(x0, v0)[0]
    P_inf_stored.append(P_gen0 * gen_s_n / sys_s_n)
    Q_inf_stored.append(Q_gen0 * gen_s_n / sys_s_n)

    # Simulation loop
    while t < t_end:
        sys.stdout.write("\r%d%%" % int(t / t_end * 100))

        # Short circuit (apply at UIC terminal bus)
        if run_sc and t_sc <= t <= (t_sc + t_sc_dur):
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = y_sc
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        sol.step()
        x = sol.x
        t = sol.t
        v = ps.solve_algebraic(t, x)

        # Step the FMU after the network/DAE step (more stable explicit coupling).
        for mdl in fmu_models:
            mdl.step_fmu(x, v, t, dt)

        result_dict['Global', 't'].append(t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        P_e_uic = uic_model.p_e(x, v)[0]
        P_ref_uic = uic_model.p_ref(x, v)[0]
        P_e_uic_pu_stored.append(P_e_uic)
        P_ref_uic_pu_stored.append(P_ref_uic)
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
        P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

        if fmu_mdl and hasattr(fmu_mdl, 'get_all_fmu_outputs'):
            d = fmu_mdl.get_all_fmu_outputs()
            # Keep FMU-reported time, but align exported Time with TOPS time vector.
            if 'Time' in d:
                d = dict(d)
                d['Time_fmu'] = d.get('Time')
                d['Time'] = float(t)
            fmu_outputs_stored.append(d)

        # Store last commanded electrical torque (as computed by the wrapper during this step)
        if fmu_mdl is not None and hasattr(fmu_mdl, '_Te_pu_cmd'):
            te_pu = float(fmu_mdl._Te_pu_cmd) if fmu_mdl._Te_pu_cmd is not None else np.nan
            Te_cmd_pu_stored.append(te_pu)
            if hasattr(fmu_mdl, '_T_base_Nm') and np.isfinite(te_pu):
                T_base_Nm = float(np.asarray(fmu_mdl._T_base_Nm).ravel()[0])
                Te_cmd_kNm_stored.append(te_pu * T_base_Nm / 1e3)
            else:
                Te_cmd_kNm_stored.append(np.nan)
        else:
            Te_cmd_pu_stored.append(np.nan)
            Te_cmd_kNm_stored.append(np.nan)
        if fmu_mdl is not None and hasattr(fmu_mdl, '_gen_spdortrq_kNm_set'):
            GenSpdOrTrq_set_kNm_stored.append(
                float(fmu_mdl._gen_spdortrq_kNm_set) if fmu_mdl._gen_spdortrq_kNm_set is not None else np.nan
            )
        else:
            GenSpdOrTrq_set_kNm_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_omega_m_pu_meas'):
            omega_m_pu_meas_stored.append(
                float(fmu_mdl._omega_m_pu_meas) if fmu_mdl._omega_m_pu_meas is not None else np.nan
            )
        else:
            omega_m_pu_meas_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_genpwr_kW_set'):
            GenPwr_set_kW_stored.append(
                float(fmu_mdl._genpwr_kW_set) if fmu_mdl._genpwr_kW_set is not None else np.nan
            )
        else:
            GenPwr_set_kW_stored.append(np.nan)

        if fmu_mdl is not None and hasattr(fmu_mdl, '_elec_pwr_com_kW_last'):
            ElecPwrCom_set_kW_stored.append(
                float(fmu_mdl._elec_pwr_com_kW_last) if fmu_mdl._elec_pwr_com_kW_last is not None else np.nan
            )
        else:
            ElecPwrCom_set_kW_stored.append(np.nan)

        X = uic_model.local_view(x)
        vi = X['vi_x'][0] + 1j * X['vi_y'][0]
        i_a = uic_model.i_a(x, v)[0]
        s_bus_actual = uic_model.s_e(x, v)[0]
        s_ref_internal = uic_model.p_ref(x, v)[0] + 1j * uic_model.q_ref(x, v)[0]
        xf = uic_model.par['xf'][0]
        s_bus_ref = s_ref_internal - 1j * xf * (np.abs(i_a) ** 2)
        P_uic_bus_actual_stored.append(s_bus_actual.real * uic_s_n / sys_s_n)
        Q_uic_bus_actual_stored.append(s_bus_actual.imag * uic_s_n / sys_s_n)
        P_uic_bus_ref_stored.append(s_bus_ref.real * uic_s_n / sys_s_n)
        Q_uic_bus_ref_stored.append(s_bus_ref.imag * uic_s_n / sys_s_n)

        v_t_uic = uic_model.v_t(x, v)[0]
        v_bus.append(np.abs(v_t_uic))
        vi_mag_hist.append(float(np.abs(vi)))
        i_a_mag_hist.append(float(np.abs(i_a)))
        i_a_angle_hist.append(float(np.angle(i_a) * 180 / np.pi))
        P_gen = gen_model.p_e(x, v)[0]
        Q_gen = gen_model.q_e(x, v)[0]
        P_inf_stored.append(P_gen * gen_s_n / sys_s_n)
        Q_inf_stored.append(Q_gen * gen_s_n / sys_s_n)

    # Terminate FMU
    for mdl in fmu_models:
        if hasattr(mdl, 'terminate_fmu'):
            mdl.terminate_fmu()

    # Convert to DataFrame and build full export
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))
    t_stored = result[('Global', 't')]

    # Build export DataFrame: power system + all FMU outputs + drivetrain states (already in result)
    omega_base_rpm_export = np.nan
    if fmu_mdl is not None and hasattr(fmu_mdl, 'par') and 'omega_m_rated' in fmu_mdl.par.dtype.names:
        omega_base_rpm_export = float(np.asarray(fmu_mdl.par['omega_m_rated']).ravel()[0])
    out_df = pd.DataFrame(
        {
            't': t_stored,
            # System pu on sys base (as plotted previously)
            'P_e_sys_pu': P_e_stored,
            'P_ref_sys_pu': P_ref_stored,
            # Raw UIC pu (signed) straight from the UIC model (matches what drivetrain sees via connection)
            'P_e_uic_pu_raw': P_e_uic_pu_stored,
            'P_ref_uic_pu_raw': P_ref_uic_pu_stored,
            'v_bus_pu': v_bus,
            'vi_mag_pu': np.asarray(vi_mag_hist, dtype=float),
            'i_a_mag_pu_uic': np.asarray(i_a_mag_hist, dtype=float),
            'i_a_angle_deg': np.asarray(i_a_angle_hist, dtype=float),
            'P_inf_sys_pu': np.asarray(P_inf_stored, dtype=float),
            'Q_inf_sys_pu': np.asarray(Q_inf_stored, dtype=float),
            'P_uic_bus_actual_sys_pu': P_uic_bus_actual_stored,
            'Q_uic_bus_actual_sys_pu': Q_uic_bus_actual_stored,
            'P_uic_bus_ref_sys_pu': P_uic_bus_ref_stored,
            'Q_uic_bus_ref_sys_pu': Q_uic_bus_ref_stored,
            # Speed base for converting FMU rpm signals to pu
            'omega_base_rpm': omega_base_rpm_export,
        }
    )
    if fmu_outputs_stored:
        df_fmu = pd.DataFrame(fmu_outputs_stored)
        out_df = pd.concat([out_df, df_fmu], axis=1)

    oe_key = ("FMUtoUICdrivetrain1", "omega_e")
    if oe_key in result.columns:
        out_df["omega_e_tops_pu"] = result[oe_key].to_numpy(dtype=float)

    # Add torque command traces (same length as t_stored)
    out_df['Te_cmd_pu'] = np.asarray(Te_cmd_pu_stored, dtype=float)
    out_df['Te_cmd_kNm'] = np.asarray(Te_cmd_kNm_stored, dtype=float)
    out_df['GenSpdOrTrq_set_kNm'] = np.asarray(GenSpdOrTrq_set_kNm_stored, dtype=float)
    out_df['omega_m_pu_meas'] = np.asarray(omega_m_pu_meas_stored, dtype=float)
    out_df['GenPwr_set_kW'] = np.asarray(GenPwr_set_kW_stored, dtype=float)
    out_df['ElecPwrCom_set_kW'] = np.asarray(ElecPwrCom_set_kW_stored, dtype=float)

    # Init-time readbacks (constant columns; useful to verify FMU latched init params)
    if fmu_mdl is not None:
        if hasattr(fmu_mdl, '_mode_write'):
            out_df['Mode_write'] = float(fmu_mdl._mode_write)
        if hasattr(fmu_mdl, '_mode_readback'):
            out_df['Mode_readback'] = float(fmu_mdl._mode_readback)
        if hasattr(fmu_mdl, '_testNr_write'):
            out_df['testNr_write'] = float(fmu_mdl._testNr_write)
        if hasattr(fmu_mdl, '_testNr_readback'):
            out_df['testNr_readback'] = float(fmu_mdl._testNr_readback)

    # Optional: implied commanded mechanical power based on FMU-reported GenSpeed (kW) and Te_cmd_kNm
    # (kN·m * rad/s = kW)
    if 'GenSpeed' in out_df.columns:
        omega_gen = out_df['GenSpeed'].to_numpy(dtype=float) * 2.0 * np.pi / 60.0
        out_df['P_cmd_kW'] = out_df['Te_cmd_kNm'].to_numpy(dtype=float) * omega_gen
        out_df['P_cmd_pu'] = out_df['P_cmd_kW'].to_numpy(dtype=float) / (uic_s_n * 1e3)
    # Logging directory (single artifact per run; always overwrite).
    ensure_log_dir(FMU_DRIVETRAIN_CSV)
    out_df.to_csv(FMU_DRIVETRAIN_CSV, index=False)
    print(f"\nResults saved to {FMU_DRIVETRAIN_CSV} ({len(out_df.columns)} columns)")

    df = pd.DataFrame(fmu_outputs_stored) if fmu_outputs_stored else None
    omega_base_rpm = None
    if fmu_mdl is not None and hasattr(fmu_mdl, 'par') and 'omega_m_rated' in fmu_mdl.par.dtype.names:
        omega_base_rpm = float(np.asarray(fmu_mdl.par['omega_m_rated']).ravel()[0])

    if not _cli.no_thesis_plots:
        thesis_dir = fmu_drivetrain_thesis_plots_dir()
        paths = save_coupled_thesis_plots(
            str(thesis_dir),
            out_df['t'].to_numpy(dtype=float),
            out_df,
            result_df=result,
            df_fmu=df,
            omega_base_rpm=omega_base_rpm,
            show=_cli.show,
        )
        print(f"Thesis figures saved to {thesis_dir} ({len(paths)} files)")

    print(f"\nSimulation took {time.perf_counter() - t_start_wall:.2f} seconds.")
    if _cli.show and not _cli.no_thesis_plots:
        import matplotlib.pyplot as plt

        plt.show(block=True)