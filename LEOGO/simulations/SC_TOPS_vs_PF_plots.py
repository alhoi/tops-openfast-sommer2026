import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import pandas as pd
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

if __name__ == '__main__':

    # Load model
    import LEOGO.LEOGO_ps as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    x_0 = ps.x_0.copy()
    t_end = 10

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()
    trip_time = 1
    clear_time = trip_time + 0.1
    sc_gen = 2
    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][sc_gen]

    event_flag = True

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t >= trip_time and t <= clear_time:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1/(0.1+1j*0.1)
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['v'].append(v.copy())
        res['gen_p'].append(ps.gen['GEN'].p_e(x, v).copy())
        res['gen_q'].append(ps.gen['GEN'].q_e(x, v).copy())

    res['gen_p'] = [arr * int(model['generators']['GEN'][1][2]) for arr in res['gen_p']] # pu to MW
    res['gen_q'] = [arr * int(model['generators']['GEN'][1][2]) for arr in res['gen_q']]
    #res['gen_speed'] = [arr + 1 for arr in res['gen_speed']]  

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))


    """ Voltages at buses """
    plot_buses_tops = ['Main Bus A', 'Utility690 Bus B', 'Drilling AC Bus B']
    plot_buses_PF = ['Main SWBD\\Main Bus B', 'Utility690 SWBD\\Utility690 Bus B', 'Drilling AC SWBD\\Drilling AC Bus B']
    idx_of_plot_buses_tops = [i for i, m in enumerate([bus[0] for bus in model['buses']][1:]) if m in plot_buses_tops]
    
    file_path_v = 'LEOGO\PF_files\SC_lines_v.csv' 
    df_raw_gen_v = pd.read_csv(file_path_v, header=None)
    labels_gen_v = df_raw_gen_v.iloc[0, 1:].tolist()
    units_gen_v = df_raw_gen_v.iloc[1, :].tolist()
    data_gen_v = df_raw_gen_v.iloc[2:].reset_index(drop=True)
    data_gen_v = data_gen_v.apply(pd.to_numeric)
    time = data_gen_v.iloc[:, 0]
    
    idx_of_plot_buses_PF = [i for i, el in enumerate(labels_gen_v) if el in plot_buses_PF]
    
    plt.figure()
    for idx in idx_of_plot_buses_tops:
        plt.plot(res['t'], [abs(v_i[idx]) for v_i in res['v']], label=f"{model['buses'][idx+1][0]} TOPS")
    for i, name in enumerate(labels for labels in labels_gen_v if labels in plot_buses_PF):
        plt.plot(time, data_gen_v.iloc[:, idx_of_plot_buses_PF[i]+1], label=name + ' PF')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [p.u.]')
    plt.title(f'Voltages at buses during SC at t={trip_time}s to t={clear_time}s')
    plt.legend()
    #plt.legend([f"{bus} TOPS" for bus in plot_buses_tops] + [f"{buz} PF" for buz in [labels_gen_v[i] for i in idx_of_plot_buses_PF]])
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.savefig(f'LEOGO\Plots\SC_line_v_comparison', dpi=300, bbox_inches='tight')
    plt.show()


    """ Active power of generators """
    file_path_gen_P = 'LEOGO\PF_files\SC_gens_P.csv'
    df_raw_gen_P = pd.read_csv(file_path_gen_P, header=None)
    labels_gen_P = df_raw_gen_P.iloc[0, 1:].tolist()
    units_gen_P = df_raw_gen_P.iloc[1, :].tolist()
    data_gen_P = df_raw_gen_P.iloc[2:].reset_index(drop=True)
    data_gen_P = data_gen_P.apply(pd.to_numeric)
    time = data_gen_P.iloc[:, 0]
    plt.figure()
    for i in range(len(res['gen_p'][0])):
        plt.plot(res['t'], [row[i] for row in res['gen_p']], label=model['generators']['GEN'][1:][i][0] + ' TOPS')
    #plt.plot(res['t'], res['gen_p'], label=labels_gen_P)
    for i, name in enumerate(labels_gen_P):
        plt.plot(time, data_gen_P.iloc[:, i+1], label=name + ' PF')
    plt.xlabel('Time [s]')
    plt.ylabel('Active power [MW]')
    plt.title(f'Active power of generators during SC at t={trip_time}s to t={clear_time}s')
    plt.legend() #[f"{gens[0]} TOPS" for gens in model['generators']['GEN'][1:]] + [f"{name} PF" for name in labels_gen_P])
    plt.ticklabel_format(useOffset=False)   
    plt.grid()
    plt.savefig(f'LEOGO\Plots\SC_gen_P_comparison', dpi=300, bbox_inches='tight')
    plt.show()


    """ Reactive power of generators """
    file_path_gen_Q = 'LEOGO\PF_files\SC_gens_Q.csv'
    df_raw_gen_Q = pd.read_csv(file_path_gen_Q, header=None)
    labels_gen_Q = df_raw_gen_Q.iloc[0, 1:].tolist()
    units_gen_Q = df_raw_gen_Q.iloc[1, :].tolist()
    data_gen_Q = df_raw_gen_Q.iloc[2:].reset_index(drop=True)
    data_gen_Q = data_gen_Q.apply(pd.to_numeric)
    time = data_gen_Q.iloc[:, 0]
    plt.figure()
    for i in range(len(res['gen_q'][0])):
        plt.plot(res['t'], [row[i] for row in res['gen_q']], label=model['generators']['GEN'][1:][i][0] + ' TOPS')
    for i, name in enumerate(labels_gen_Q):
        plt.plot(time, data_gen_Q.iloc[:, i+1], label=name + ' PF')
    plt.xlabel('Time [s]')
    plt.ylabel('Reactive power [MVAr]')
    plt.title(f'Reactive power of generators during SC at t={trip_time}s to t={clear_time}s')
    plt.legend() #[f"{gens[0]} TOPS" for gens in model['generators']['GEN'][1:]] + [f"{name} PF" for name in labels_gen_Q])
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.savefig(f'LEOGO\Plots\SC_gen_Q_comparison', dpi=300, bbox_inches='tight')
    plt.show()


    """ Generator speed """
    file_path_gen_speed = 'LEOGO\PF_files\SC_gens_speed.csv'
    df_raw_gen_speed = pd.read_csv(file_path_gen_speed, header=None)
    labels_gen_speed = df_raw_gen_speed.iloc[0, 1:].tolist()
    units_gen_speed = df_raw_gen_speed.iloc[1, :].tolist()
    data_gen_speed = df_raw_gen_speed.iloc[2:].reset_index(drop=True)
    data_gen_speed = data_gen_speed.apply(pd.to_numeric)
    data_gen_speed.iloc[:, 1:] = data_gen_speed.iloc[:, 1:] - 1 # adjusting speed to relative
    time = data_gen_speed.iloc[:, 0]
    plt.figure()
    for i in range(len(res['gen_speed'][0])):
        plt.plot(res['t'], [row[i] for row in res['gen_speed']], label=model['generators']['GEN'][1:][i][0] + ' TOPS')
    for i, name in enumerate(labels_gen_speed):
        plt.plot(time, data_gen_speed.iloc[:, i+1], label=name + ' PF')
    plt.xlabel('Time [s]')
    plt.ylabel('Gen. speed (relative)')
    plt.title(f"Generator speed during SC at t={trip_time}s to t={clear_time}s")
    plt.legend() #[f"Gen {i+1} TOPS" for i in range(len(res['gen_speed'][0]))] + [f"{labels_gen_speed} PF" for labels_gen_speed in labels_gen_speed])
    plt.grid()
    plt.savefig(f'LEOGO\Plots\SC_gen_speed_comparison', dpi=300, bbox_inches='tight')
    plt.show()

    