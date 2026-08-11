from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
from tops_openfast.anim import Player

# region Model loading and initialisation stage
from casestudies.ps_data import assignment_model as model_data
model = model_data.load()
ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

ps.power_flow()  # Power flow calculation
ps.init_dyn_sim()  # Initialise dynamic variables
x0 = ps.x0.copy()  # Initial states

t = 0
t_end = 5  # Simulation time
maxstep = 2e-3

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
E_f_stored = []
v_stored = []
e_st_stored = []
e_t_stored = []
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
    P_m_stored.append(ps.gen['GEN'].P_m(x, v).copy())
    P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
    E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())
    e_st_stored.append(ps.gen['GEN'].e_st(x, v).copy())
    e_t_stored.append(ps.gen['GEN'].e_t(x, v).copy())
    v_stored.append(ps.gen['GEN'].v_t(x, v).copy())
    I_gen = ps.y_bus_red_full[0, 1] * (v[0] - v[1])
    I_stored.append(np.abs(I_gen))
    # endregion

# Convert dict to pandas dataframe
result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

## Here comes animation

oscillators = np.hstack((np.array(E_f_stored),np.array(e_st_stored),np.array(e_t_stored),np.array(v_stored)))

def update(frame):
    # for each frame, update the data stored on each artist.
    x, y = oscillators[frame].real, oscillators[frame].imag
    # update the scatter plot:
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    # update the line plot:

    return scat


fig, ax = plt.subplots()
ngens = 2
clrs = ['blue']*ngens+['red']*ngens+['green']*ngens +['black']*ngens
#clrs = ['blue','blue','red','red','green','green']
circle = plt.Circle((0, 0), 1, color='r', fill=False)
ax.add_patch(circle)
scat = ax.scatter(oscillators[0].real,oscillators[0].real, s=10, color = clrs)
#line2 = ax.plot(time[0], 1, alpha=0.3)[0]
ax.set(xlim=[-1.2, 1.5], ylim=[-0.5, 1.6])

ani = Player(fig=fig, func=update, frames=len(v_stored), interval=2, maxi=len(oscillators)-1)
plt.show()
