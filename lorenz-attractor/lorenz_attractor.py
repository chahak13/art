import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def ode_model(t, state, kwargs):
    x, y, z = state
    return [
        kwargs['sigma'] * (y - x),
        x * (kwargs['rho'] - z),
        x * y - kwargs['beta'] * z
    ]

variables = {
    'rho': 28.0,
    'sigma': 10.0,
    'beta': 8/3
}

initial_state = [1.0, 1.0, 1.0]
t_start, t_end = 0, 40
t = np.arange(t_start, t_end, 0.01)

sol = solve_ivp(ode_model, (t_start, t_end), initial_state, t_eval = t, dense_output=True, args=(variables,))

print(sol['y'].shape)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
# ln, = ax.plot([], [], 'o', markersize=1)
minx, miny, minz = np.min(sol['y'], axis=1)
maxx, maxy, maxz = np.max(sol['y'], axis=1)

ax.plot(sol['y'][0, :], sol['y'][2, :], 'k', linewidth=0.5, alpha=0.3)
ax.set_xlim(minx, maxx)
ax.set_ylim(minz, maxz)
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
fig.tight_layout()
mark, = ax.plot(sol['y'][0, 0], sol['y'][2, 0], 'ro', markersize=2.5)
def update(frame):
    # if frame > 0:
        # ax.plot(sol['y'][0, :frame-1], sol['y'][1, :frame-1], 'ko', markersize=1)
    mark.set_data(sol['y'][0, frame], sol['y'][2, frame])

ani = FuncAnimation(fig, update, interval=34, frames=len(t)//5)
# plt.show()
ani.save('lorenz-black.mp4', writer='ffmpeg')
