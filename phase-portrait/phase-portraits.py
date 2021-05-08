import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import product
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

def ode(t, z):
    x, y = z
    return [-x + x**3, -2*y]
    # return [-y - 0.5*x*(x**2 + y**2), x - 0.5*y*(x**2 + y**2)]
    # return [y - y**3, -x-y**2]
    # return [-2*np.cos(x)-np.cos(y), -2*np.cos(y)-np.cos(x)]

t = np.linspace(1, 8, 1000)
x_space, y_space = np.linspace(-np.pi/2 - 1, np.pi/2+1, 20), np.linspace(-np.pi/2-1, np.pi/2+1, 20)
solutions = []

for x0, y0 in product(x_space, y_space):
    sol = solve_ivp(ode, (1, 20), (x0, y0), t_eval=t)
    solutions.append(sol['y'])

fig_anim, ax_anim = plt.subplots()
ax_anim.set_xticks([])
ax_anim.set_yticks([])
ax_anim.set_xlim(-1.75, 1.75)
ax_anim.set_ylim(-2, 2)

def update(frame):
    ax_anim.collections = []
    if frame < 140:
        length = [int(0.1*sol.shape[1]) for sol in solutions]
        xframe = [sol[0, :frame+l] for l, sol in zip(length, solutions)]
        yframe = [sol[1, :frame+l] for l, sol in zip(length, solutions)]
        stack = [np.column_stack([x, y]) for x, y in zip(xframe, yframe)]
        segments = LineCollection(stack, color='k', linewidth=0.7)
        ax_anim.add_collection(segments)
    return

anim = FuncAnimation(fig_anim, update, frames=150)
# anim.save('phase.mp4', writer='ffmpeg', fps=30)
anim.save('phase-small.gif', writer='imagemagick', fps=30)
