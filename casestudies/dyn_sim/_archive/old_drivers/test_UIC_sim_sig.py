import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import sys


if __name__ == '__main__':

    # region Model loading and initialisation stage
    import casestudies.ps_data.uic_ib_sig as model_data
    model_name = 'uic_ib_sig'  # Name of the model data file
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object
    ps.power_flow()  # Power flow calculation
    ps.init_dyn_sim()  # Initialise dynamic variables
    
    ps.gen['GEN'].sys_par = ps.sys_data.copy()
    
    x0 = ps.x0.copy()  # Initial states

    t = 0
    t_freq_start = 5  # Time when frequency disturbance starts
    gen_speed_idx = ps.gen['GEN'].state_idx_global['speed'][0]  # IB speed index
    result_dict = defaultdict(list)
    t_end = 20  # Simulation time

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)
    print('Largest mismatch: ', max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))
    print(ps.state_derivatives(0, ps.x_0, ps.v_0))
    # endregion

    # region Runtime variables
    I_a_stored = []
    theta_stored = []
    # Power references (external commands, from model parameters)
    p_ref_stored = []
    q_ref_stored = []
    v_ref_stored = []
    vi_mag_stored = []
    P_actual_stored = []
    Q_actual_stored = []

    # change these for events
    short_circuit_flag = False 
    freq_response_flag = True 

    perfect_tracking = ps.vsc['UIC_sig'].par['perfect_tracking']
    frequency_change = False
    sc_bus_idx = ps.vsc['UIC_sig'].bus_idx_red['terminal'][0]
    # endregion

    # Simulation loop starts here!
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        # Update dt in the VSC model for accurate omega calculation
        ps.vsc['UIC_sig'].dt = sol.dt
        
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t
        
        # Apply frequency disturbance by modifying generator speed
        if t >= t_freq_start and freq_response_flag and not frequency_change:
            x[gen_speed_idx] += -0.001
            frequency_change = True
        
        # Short circuit
        if 15 <= t <= 15.05 and short_circuit_flag:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e5
        else:
            ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

        I_a = ps.vsc['UIC_sig'].i_a(x, v)
        I_a_abs = np.abs(I_a)
        I_a_stored.append(I_a_abs)
        
        vsc_local = ps.vsc['UIC_sig'].local_view(x)
        vi_x = vsc_local['vi_x']
        vi_y = vsc_local['vi_y']
        vi = vi_x + 1j*vi_y
        theta = vi / abs(vi) if abs(vi) > 1e-10 else 0  
        theta_stored.append(np.angle(theta)) 
        
        vi_mag_stored.append(abs(vi))

        # Use original external power references from model parameters (do not modify with xf)
        p_ref_cmd = np.atleast_1d(ps.vsc['UIC_sig'].par['p_ref'])[0]
        q_ref_cmd = np.atleast_1d(ps.vsc['UIC_sig'].par['q_ref'])[0]
        p_ref_stored.append(p_ref_cmd)
        q_ref_stored.append(q_ref_cmd)

        v_ref_stored.append(ps.vsc['UIC_sig']._input_values['v_ref'])
        
        s_actual = ps.vsc['UIC_sig'].s_e(x, v)
        P_actual_stored.append(s_actual.real)
        Q_actual_stored.append(s_actual.imag)
        # endregion

    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

    # region Plotting
    t_stored = result[('Global', 't')]

    # Determine plot title based on events and perfect tracking
    perfect_tracking_enabled = perfect_tracking.any() if hasattr(perfect_tracking, 'any') else bool(perfect_tracking)
    if freq_response_flag:
        if perfect_tracking_enabled:
            plot_title = 'UIC response to frequency event with perfect tracking'
        else:
            plot_title = 'UIC response to frequency event with droop control'
    else:
        plot_title = 'UIC steady state'

    fig, ax = plt.subplots(3)
    fig.suptitle(plot_title)
    ax[0].plot(t_stored, result.xs(key='vi_x', axis='columns', level=1), label='vi_x', color='#FF1493')  # deeppink
    ax[0].plot(t_stored, result.xs(key='vi_y', axis='columns', level=1), label='vi_y', color='blue')
    ax[0].legend()
    ax[0].set_ylabel('Controller states')
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    
    ax[1].plot(t_stored, np.array(I_a_stored), label='I_a', color='#FF1493')  # deeppink
    ax[1].legend()
    ax[1].set_ylabel('VSC Current (pu)')
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    
    ax[2].plot(t_stored, np.array(theta_stored)*180/np.pi, label='theta', color='#FF1493')  # deeppink
    ax[2].legend()
    ax[2].set_ylabel('Voltage angle (deg)')
    ax[2].set_xlabel('Time (s)')
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    # Layout for first figure
    plt.tight_layout()
    
    # Create second figure with references (only displayed, not saved)
    fig2, ax2 = plt.subplots(3, figsize=(10, 8))
    fig2.suptitle(plot_title)
    
    # Plot 1: vi magnitude vs v_ref
    ax2[0].plot(t_stored, np.array(vi_mag_stored), label='|vi| actual', linewidth=2, color='#FF1493')  # deeppink
    ax2[0].plot(t_stored, np.array(v_ref_stored), '--', label='v_ref', linewidth=2, color='blue')
    ax2[0].legend()
    ax2[0].set_ylabel('Voltage magnitude (pu)')
    ax2[0].grid(True)
    ax2[0].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    
    # Plot 2: P vs p_ref (external command)
    ax2[1].plot(t_stored, np.array(P_actual_stored), label='P actual', linewidth=2, color='#FF1493')  # deeppink
    ax2[1].plot(t_stored, np.array(p_ref_stored), '--', label='P ref (command)', linewidth=2, color='blue')
    ax2[1].legend()
    ax2[1].set_ylabel('Active power (pu)')
    ax2[1].grid(True)
    ax2[1].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    
    # Plot 3: Q vs q_ref (external command)
    ax2[2].plot(t_stored, np.array(Q_actual_stored), label='Q actual', linewidth=2, color='#FF1493')  # deeppink
    ax2[2].plot(t_stored, np.array(q_ref_stored), '--', label='Q ref (command)', linewidth=2, color='blue')
    ax2[2].legend()
    ax2[2].set_ylabel('Reactive power (pu)')
    ax2[2].set_xlabel('Time (s)')
    ax2[2].grid(True)
    ax2[2].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    
    plt.tight_layout()
    plt.show()
    # endregion
