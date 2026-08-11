import sys
import tops.dynamic as dps
from tops.simulator import Simulator
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

if __name__ == '__main__':

    import LEOGO.LEOGO_ps as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    #ps.power_flow()
    ps.init_dyn_sim()
    
    ps.ode_fun(0, ps.x0)
    sim = Simulator(ps, dt=5e-3, t_end=2)
    res = []
    #sim.interface_functions['print'] = lambda sim: print(f'Time: {sim.sol.t}, State: {sim.sol.y}')

    
    #print(sim.ps.state_desc, '\n', sim.sol.x)
    #sim.sol.v
    #print(f'Lines: {sim.ps.lines["Line"]}')
    print(f'p_e: {sim.ps.gen["GEN"].p_e(sim.sol.x, sim.sol.v)}')
    print(f'q_e: {sim.ps.gen["GEN"].q_e(sim.sol.x, sim.sol.v)}')
    load_idx, p_load, q_load = sim.ps.loads['Load'].load_flow_pq()
    #print(load_idx, p_load, q_load)
    
    #print(f"loads: {sim.ps.loads['Load'].load_flow_pq()}")
    """ loads = model['loads']
    print(f"Loads: {loads['Load']}") """
    buses = model['buses']
    #for i, (bus, vol) in enumerate(buses): print(i, bus)
    """ for i, idx in enumerate(load_idx):
        load_bus = buses[idx+1][0]
        print(f'Bus {load_bus} with index {idx} has P: {p_load[i]} and Q: {q_load[i]}') """
    #[print(f'Bus: {idx}, P: {p_load[i]}, Q: {q_load[i]}') for i, idx in enumerate(load_idx)]

    sim.main_loop()
    #print(f'Voltages:\n{sim.sol.v}')
    """ for i, v in enumerate(sim.sol.v):
        bus_name = buses[i+1][0]
        print(f'Bus {i}: {bus_name}: {abs(v):.3f} p.u.') """

    print(f'Voltage magnitudes:\n{abs(sim.sol.v)}')
    
    print('Done')
