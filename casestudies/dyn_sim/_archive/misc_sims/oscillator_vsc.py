from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)

# region Model loading and initialisation stage
from casestudies.ps_data import one_vsc_ib as model_data
model = model_data.load()
ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

ps.power_flow()  # Power flow calculation
ps.init_dyn_sim()  # Initialise dynamic variables
x0 = ps.x0.copy()  # Initial states

t = 0
t_end = 5  # Simulation time
maxstep = 1e-3

# Solver
sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)
# endregion

# Print load flow solution
for bus, v in zip(ps.buses['name'], ps.v_0):
    print(f'{bus}: {np.abs(v):.2f} /_ {np.angle(v):.2f}')

# region Runtime variables
time = []
result_dict = defaultdict(list)
# Additional plot variables
P_m_stored = []
P_e_stored = []
Q_e_stored = []
v_q_stored = []
v_bus = []
v_stored = []
I_stored = []
modal_stored = []

event_flag1 = True
# endregion
sc_idx = 0
# Simulation loop starts here!
while t < t_end:
    result = sol.step()
    x = sol.y
    v = sol.v
    t = sol.t

    if 1<t<1.20:
        ps.y_bus_red_mod[sc_idx,sc_idx] = 1e6
    else:
        ps.y_bus_red_mod[sc_idx,sc_idx] = 0

    # region Store variables
    time.append(sol.t)
    [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]

    # Store additional variables
    P_e_stored.append(ps.vsc['VSC_PV'].p_e(x, v).copy())
    Q_e_stored.append(ps.vsc['VSC_PV'].q_e(x, v).copy())
    v_q_stored.append(ps.vsc['VSC_PV'].v_q(x, v).copy())
    v_bus.append(ps.vsc['VSC_PV'].v_t(x, v).copy())
    I_vsc = ps.y_bus_red_full[0, 1] * (v[0] - v[1])
    I_stored.append(ps.vsc['VSC_PV'].i_inj(x, v))
    v_stored.append(ps.gen['GEN'].v_t(x, v).copy())
    # endregion

# Convert dict to pandas dataframe
result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

## Here comes animation

from tops_openfast.anim import Player
oscillators = np.hstack((np.array(I_stored),np.array(v_stored),np.array(v_bus)))

def update(frame):
    # for each frame, update the data stored on each artist.
    x, y = oscillators[frame].real, oscillators[frame].imag
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:

    return scat


fig, ax = plt.subplots()
ngens = 1
clrs = ['blue']*ngens+['red']*ngens+['green']*ngens #+['black']*ngens
#clrs = ['blue','blue','red','red','green','green']
circle = plt.Circle((0, 0), 1, color='r', fill=False)
ax.add_patch(circle)
scat = ax.scatter(oscillators[0].real,oscillators[0].real, s=10, color = clrs)
#line2 = ax.plot(time[0], 1, alpha=0.3)[0]
ax.set(xlim=[-1.2, 1.5], ylim=[-1.3, 1.6])

ani = Player(fig=fig, func=update, frames=len(v_stored), interval=2, maxi=len(oscillators)-1)
plt.show()