import sys
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)

project_root = Path(__file__).parents[2]
# if project_root not in sys.path:
    # sys.path.insert(0, str(project_root))
# Like DynaWind: cwd = project root so extract(unzipdir='openfast_fmu') and wd path work
os.chdir(project_root)

from casestudies.dyn_sim.plotting.log_paths import FMU_UIC_CSV, ensure_log_dir

if __name__ == '__main__':
    # Model loading and initialisation
    import casestudies.ps_data.test_WT_FMU_ as model_data
    model = model_data.load()

    import tops_openfast.dyn_models as ext_lib
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=ext_lib)  # Load into a PowerSystemModel object


    # UIC p_ref for power flow (FMU provides it during dynamics via connection)
    uic_model = ps.vsc['UIC_sig']
    uic_model.par['p_ref'][:] = 0.5
    uic_model.par['q_ref'][:] = 0.0

    ps.power_flow()
    ps.init_dyn_sim()
    x0 = ps.x0.copy()
    v0 = ps.v0.copy()

    fmu_models = [mdl for mdl in ps.dyn_mdls if hasattr(mdl, 'step_fmu')]
    gen_model = ps.gen['GEN']
    uic_name = uic_model.par['name'][0]
    gen_name = gen_model.par['name'][0]

    t = 0
    result_dict = defaultdict(list)
    # t_end = 120
    t_end = 20
    dt = 0.01
    # Use dt=0.01 to match OpenFAST FMU (canHandleVariableCommunicationStepSize=false)
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=dt)

    # Runtime storage (all FMU outputs from modelDescription.xml)
    P_e_stored = []
    P_ref_stored = []
    v_bus = []
    fmu_outputs_stored = []  # list of dicts, one per step

    # Initial point
    v0 = ps.solve_algebraic(0, x0)
    result_dict['Global', 't'].append(0)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x0)]
    sys_s_n = ps.sys_data['s_n']
    uic_s_n = uic_model.par['S_n'][0]
    gen_s_n = gen_model.par['S_n'][0]

    P_e_uic = uic_model.p_e(x0, v0)[0]
    P_ref_uic = uic_model.p_ref(x0, v0)[0]
    P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
    P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

    fmu_mdl = fmu_models[0] if fmu_models else None
    if fmu_mdl and hasattr(fmu_mdl, 'get_all_fmu_outputs'):
        fmu_outputs_stored.append(fmu_mdl.get_all_fmu_outputs())

    v_t_uic = uic_model.v_t(x0, v0)[0]
    v_bus.append(np.abs(v_t_uic))

    # Simulation loop
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        for mdl in fmu_models:
            mdl.step_fmu(x, v, t, dt)

        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        P_e_uic = uic_model.p_e(x, v)[0]
        P_ref_uic = uic_model.p_ref(x, v)[0]
        P_e_stored.append(P_e_uic * uic_s_n / sys_s_n)
        P_ref_stored.append(P_ref_uic * uic_s_n / sys_s_n)

        if fmu_mdl and hasattr(fmu_mdl, 'get_all_fmu_outputs'):
            fmu_outputs_stored.append(fmu_mdl.get_all_fmu_outputs())

        v_t_uic = uic_model.v_t(x, v)[0]
        v_bus.append(np.abs(v_t_uic))

    # Terminate FMU
    for mdl in fmu_models:
        if hasattr(mdl, 'terminate_fmu'):
            mdl.terminate_fmu()

    # Convert to DataFrame and build full export
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))
    t_stored = result[('Global', 't')]

    # Build export DataFrame: power system + all FMU outputs
    out_df = pd.DataFrame({'t': t_stored, 'P_e_pu': P_e_stored, 'P_ref_pu': P_ref_stored, 'v_bus_pu': v_bus})
    if fmu_outputs_stored:
        df_fmu = pd.DataFrame(fmu_outputs_stored)
        out_df = pd.concat([out_df, df_fmu], axis=1)
    ensure_log_dir(FMU_UIC_CSV)
    out_df.to_csv(FMU_UIC_CSV, index=False)
    print(f"\nResults saved to {FMU_UIC_CSV} ({len(out_df.columns)} columns)")

    # Plot in windows with 2–3 related subplots each (colors match uic_sim)
    PLOT_COLORS = ['blue', '#FF1493', 'orange', 'green']  # deeppink, same as uic_sim
    df = pd.DataFrame(fmu_outputs_stored) if fmu_outputs_stored else None

    # Window 1: Power system and wind (3 subplots)
    n_power = 3 if df is not None and any(c in df.columns for c in ['Wind1VelX', 'RtVAvgxh']) else 2
    fig1, axes1 = plt.subplots(n_power, 1, sharex=True, figsize=(9, 4 * n_power))
    axes1 = np.atleast_1d(axes1)
    fig1.suptitle('Power system & wind', fontsize=12)
    axes1[0].plot(t_stored, P_e_stored, label='P_e (UIC)', color=PLOT_COLORS[1], linewidth=1.5)
    axes1[0].plot(t_stored, P_ref_stored, '--', label='P_ref (FMU→UIC)', color=PLOT_COLORS[0], linewidth=1.5)
    axes1[0].set_ylabel('Power (p.u.)')
    axes1[0].legend(loc='best', fontsize=8)
    axes1[0].grid(True, alpha=0.3)
    axes1[1].plot(t_stored, v_bus, label='|v_t| UIC terminal', color=PLOT_COLORS[0], linewidth=1.5)
    axes1[1].set_ylabel('UIC terminal voltage |v_t| (p.u.)')
    axes1[1].legend(loc='best', fontsize=8)
    axes1[1].grid(True, alpha=0.3)
    axes1[1].yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))
    axes1[1].ticklabel_format(style='plain', axis='y')
    if n_power == 3:
        for i, col in enumerate(['Wind1VelX', 'RtVAvgxh']):
            if col in df.columns:
                axes1[2].plot(t_stored, df[col], label=col, color=PLOT_COLORS[i % len(PLOT_COLORS)], linewidth=1.5)
        axes1[2].set_ylabel('Wind (m/s)')
        axes1[2].set_xlabel('Time (s)')
        axes1[2].legend(loc='best', fontsize=8)
        axes1[2].grid(True, alpha=0.3)
    else:
        axes1[1].set_xlabel('Time (s)')
    fig1.tight_layout()

    if df is not None:
        # Window 2: Drivetrain (torque and speed) – 2 subplots
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        fig2.suptitle('Drivetrain', fontsize=12)
        for i, col in enumerate(['HSShftTq', 'GenTq']):
            if col in df.columns:
                ax2a.plot(t_stored, df[col], label=col, color=PLOT_COLORS[i % len(PLOT_COLORS)], linewidth=1.5)
        ax2a.set_ylabel('Torque (kN-m)')
        ax2a.legend(loc='best', fontsize=8)
        ax2a.grid(True, alpha=0.3)
        for i, col in enumerate(['GenSpeed', 'RotSpeed']):
            if col in df.columns:
                ax2b.plot(t_stored, df[col], label=col, color=PLOT_COLORS[i % len(PLOT_COLORS)], linewidth=1.5)
        ax2b.set_ylabel('Speed (rpm)')
        ax2b.set_xlabel('Time (s)')
        ax2b.legend(loc='best', fontsize=8)
        ax2b.grid(True, alpha=0.3)
        fig2.tight_layout()

        # Window 3: Angles (blade pitch, nacelle yaw, rotor position) – 3 subplots
        fig3, (ax3a, ax3b, ax3c) = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
        fig3.suptitle('Angles', fontsize=12)
        if 'BldPitch1' in df.columns:
            ax3a.plot(t_stored, df['BldPitch1'], label='BldPitch1', color=PLOT_COLORS[0], linewidth=1.5)
        ax3a.set_ylabel('Blade pitch (deg)')
        ax3a.legend(loc='best', fontsize=8)
        ax3a.grid(True, alpha=0.3)
        if 'NacYaw' in df.columns:
            ax3b.plot(t_stored, df['NacYaw'], label='NacYaw', color=PLOT_COLORS[0], linewidth=1.5)
        ax3b.set_ylabel('Nacelle yaw (deg)')
        ax3b.legend(loc='best', fontsize=8)
        ax3b.grid(True, alpha=0.3)
        for i, col in enumerate(['Azimuth', 'LSSGagPxa']):
            if col in df.columns:
                ax3c.plot(t_stored, df[col], label=col, color=PLOT_COLORS[i % len(PLOT_COLORS)], linewidth=1.5)
        ax3c.set_ylabel('Rotor position (deg)')
        ax3c.set_xlabel('Time (s)')
        ax3c.legend(loc='best', fontsize=8)
        ax3c.grid(True, alpha=0.3)
        fig3.tight_layout()

        # Window 4: Accelerations (drivetrain and tower) – 2 subplots
        fig4, (ax4a, ax4b) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        fig4.suptitle('Accelerations', fontsize=12)
        if 'GenAccel' in df.columns:
            ax4a.plot(t_stored, df['GenAccel'], label='GenAccel', color=PLOT_COLORS[0], linewidth=1.5)
        ax4a.set_ylabel('Gen. accel (deg/s²)')
        ax4a.legend(loc='best', fontsize=8)
        ax4a.grid(True, alpha=0.3)
        for i, col in enumerate(['YawBrTAxp', 'YawBrTAyp']):
            if col in df.columns:
                ax4b.plot(t_stored, df[col], label=col, color=PLOT_COLORS[i % len(PLOT_COLORS)], linewidth=1.5)
        ax4b.set_ylabel('Tower accel (m/s²)')
        ax4b.set_xlabel('Time (s)')
        ax4b.legend(loc='best', fontsize=8)
        ax4b.grid(True, alpha=0.3)
        fig4.tight_layout()

    plt.show()
