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
    import casestudies.ps_data.assignment_model as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

    ps.power_flow()  # Power flow calculation

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    t = 0
    result_dict = defaultdict(list)
    t_end = 10  # Simulation time

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)
    # endregion

    # region Print initial conditions
    v_bus_mag = np.abs(ps.v_0)
    v_bus_angle = np.angle(ps.v_0)  # In radians
    print(f'Voltages (pu): {v_bus_mag}')
    print(f'Voltage angles: {v_bus_angle} \n')
    print(f'state description: \n {ps.state_desc} \n')
    print(f'Initial values on all state variables (G1 and IB) : \n {x0} \n')
    # endregion

    # region Runtime variables
    # Additional plot variables
    P_m_stored = []
    P_e_stored = []
    E_f_stored = []
    v_bus = []
    I_stored = []

    # event_flag1 = True  # Create a different flag for each system event
    # endregion

    # Simulation loop starts here!
    while t < t_end:
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

        # Short circuit
        if 1 <= t <= 1.05:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables
        P_m_stored.append(ps.gen['GEN'].P_m(x, v).copy())
        P_e_stored.append(ps.gen['GEN'].p_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())

        I_gen = ps.y_bus_red_full[0, 1] * (v[0] - v[1])
        I_stored.append(np.abs(I_gen))  # Stores magnitude of armature current
        v_bus.append(np.abs(v[0]))  # Stores magnitude of generator terminal voltage
        # endregion

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

    # region Plotting
    t_stored = result[('Global', 't')]

    fig, ax = plt.subplots(3)
    fig.suptitle('Generator speed, power angle and electric power')
    ax[0].plot(t_stored, result.xs(key='speed', axis='columns', level=1))
    ax[1].plot(t_stored, result.xs(key='angle', axis='columns', level=1))
    ax[2].plot(t_stored, np.array(P_e_stored), label='P_e')
    ax[2].plot(t_stored, np.array(P_m_stored), label='P_m')

    ax[0].set_ylabel('Speed (p.u.)')
    ax[1].set_ylabel('Power angle (rad)')
    ax[2].set_ylabel('Active power (p.u.)')
    ax[2].set_xlabel('Time (s)')
    plt.legend()
    plt.show(block = True)
    # endregion
