from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)


if __name__ == '__main__':

    # region Model loading and initialisation stage
    case = "classic"
    case = "GFL_A1"
    # case = "GFM_A1"

    if case == "classic":
        import casestudies.ps_data.k2a_regulated as model_data
    elif case == "GFL_A1":
        import casestudies.ps_data.k2a_GFL_A1 as model_data
    elif case == "GFM_A1":
        import casestudies.ps_data.k2a_GFM_A1 as model_data
    
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

    ps.power_flow()  # Power flow calculation
    # Print load flow solution
    for bus, v in zip(ps.buses['name'], ps.v_0):
        print(f'{bus}: {np.abs(v):.2f} /_ {np.angle(v):.2f}')

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    # List of machine parameters for easy access 
    gen_pars = ps.gen['GEN'].par # Access like this: S_n_gen = genpars['S_n']

    t = 0
    t_end = 20  # Simulation time

    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)  # solver
    # endregion

    # region Runtime variables
    result_dict = defaultdict(list)
    # Additional plot variables
    P_m_stored = []
    P_e_stored = []
    E_f_stored = []
    v_bus = []
    I_stored = []
    modal_stored = []

    event_flag1 = True
    # endregion

    # Simulation loop starts here!
    while t < t_end:
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        if 3+1 < t < 3+1.05:
            ps.y_bus_red_mod[6,6] = 1e6
        else:
            ps.y_bus_red_mod[6,6] = 0

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables
        P_m_stored.append(ps.gen['GEN'].P_m(x, v).copy())
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())
        I_gen = ps.y_bus_red_full[0, 1] * (v[0] - v[1])
        I_stored.append(np.abs(I_gen))
        v_bus.append([np.abs(v[0]), np.abs(v[1]), np.abs(v[2]), np.abs(v[3])])  # Store bus voltage magnitudes
        # endregion

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))
    print(result.head()) 

    # region Print initial conditions
    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = np.angle(ps.v_0)

    # region Plotting
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('Generator speed, bus voltage and electric power')
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1), label=gen_pars['name'])
    ax[0].legend()
    ax[0].set_ylabel('Speed (p.u.)')
    ax[1].plot(result[('Global', 't')], v_bus, label=['Bus 1', 'Bus 2', 'Bus 3', 'Bus 4'])
    ax[1].legend()
    ax[1].set_ylabel('Bus voltage (p.u.)')
    ax[2].plot(result[('Global', 't')], np.array(P_e_stored)/gen_pars['S_n'], label=gen_pars['name'])
    ax[2].legend()
    ax[2].set_ylabel('Elec. power (p.u.)')
    ax[2].set_xlabel('time (s)')
    plt.show()
    # endregion