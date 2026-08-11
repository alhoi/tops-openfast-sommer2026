import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

if __name__ == '__main__':

    # Load model
    import LEOGO.LEOGO_ps as model_data
    model = model_data.load()

    """ model["avr"] = {}
    model['gov'] = {}
    model['generators']["GEN"] = model['generators']["GEN"][:2] """
    
    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 10
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()
    trip_time = 2
    clear_time = trip_time + 0.01
    sc_gen = 2

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][sc_gen]

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if t >= trip_time and t <= clear_time:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6
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
        res['load_p'].append(ps.loads['Load'].p(x, v).copy())
        res['load_q'].append(ps.loads['Load'].q(x, v).copy())

        loads = [entry[0] for entry in model['loads']]
        loads = loads[1:] 
        load_flow_loads = {load: [[], []] for load in loads}

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    plt.figure()
    plt.plot(res['t'], [abs(v_i) for v_i in res['v']])
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [p.u.]')
    plt.title(f"Voltages at buses during short circuit at {model['generators']['GEN'][sc_gen][0]} at t={trip_time}s to t={clear_time}s")
    plt.legend([buses[0] for buses in model['buses'][1:]])
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['gen_p'])
    plt.xlabel('Time [s]')
    plt.ylabel('Active power [p.u.]')
    plt.title(f"Active power of generators during short circuit at {model['generators']['GEN'][sc_gen][0]} at t={trip_time}s to t={clear_time}s")
    plt.legend([gens[0] for gens in model['generators']['GEN'][1:]])
    plt.ticklabel_format(useOffset=False)   
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['gen_q'])
    plt.xlabel('Time [s]')
    plt.ylabel('Reactive power [p.u.]')
    plt.title(f"Reactive power of generators during short circuit at {model['generators']['GEN'][sc_gen][0]} at t={trip_time}s to t={clear_time}s")
    plt.legend([gens[0] for gens in model['generators']['GEN'][1:]])
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['load_p'])
    plt.xlabel('Time [s]')
    plt.ylabel('Active power [p.u.]')
    plt.title(f"Active power of loads during short circuit at {model['generators']['GEN'][sc_gen][0]} at t={trip_time}s to t={clear_time}s")
    plt.legend([load[0] for load in model['loads'][1:]])
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['load_q'])
    plt.xlabel('Time [s]')
    plt.ylabel('Reactive power [p.u.]')
    plt.title(f"Reactive power of loads during short circuit at {model['generators']['GEN'][sc_gen][0]} at t={trip_time}s to t={clear_time}s")
    plt.legend([load[0] for load in model['loads'][1:]])
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['gen_speed'])
    plt.xlabel('Time [s]')
    plt.ylabel('Gen. speed (relative)')
    plt.title(f"Generator speed during short circuit at {model['generators']['GEN'][sc_gen][0]} at t={trip_time}s to t={clear_time}s")
    plt.legend([f"Gen {i+1}" for i in range(len(res['gen_speed'][0]))])
    plt.grid()
    plt.show()


    """ plt.figure()
    plot_buses = ['10', '13', '22', '41'] # add buses to plot here
    bus_names = [model['buses'][int(el)+1][0] for el in plot_buses]
    for el, name in zip(plot_buses, bus_names):
        plt.plot(res['t'], voltages[el], label=f'Bus {name}')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [p.u.]')
    plt.title(f'Voltages at buses during short circuit at t={trip_time}s to t={clear_time}s')
    plt.legend([el for el in bus_names])
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure()
    for g in p_gens:
        plt.plot(res['t'], p_gens[g], label=f"P Gen {int(g)+1}")
    plt.xlabel('Time [s]')  
    plt.ylabel('Active power [p.u.]')
    plt.title(f'Active power of generators during short circuit at t={trip_time}s to t={clear_time}s')
    plt.legend()
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure()
    for g in q_gens:
        plt.plot(res['t'], q_gens[g], label=f'Q Gen {int(g)+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Reactive power [p.u.]')
    plt.title(f'Reactive power of generators during short circuit at t={trip_time}s to t={clear_time}s')
    plt.legend()
    plt.ticklabel_format(useOffset=False)
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(res['t'], res['gen_speed'])
    plt.xlabel('Time [s]')
    plt.ylabel('Gen. speed')
    plt.title(f'Generator speed during line outage at t={trip_time}s')
    plt.legend([f'Gen {i+1}' for i in range(len(res['gen_speed'][0]))])
    plt.grid()
    plt.show() """