import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

def ikeda_trajectory(u, x, y, N):
    trajectory = np.zeros((x.shape[0], N, 2))
    trajectory[:, 0, 0] = x
    trajectory[:, 0, 1] = y

    for n in range(N-1):
        x_n = trajectory[:, n, 0]
        y_n = trajectory[:, n, 1]

        t_n = 0.4 - 6/(1 + x_n**2 + y_n**2)

        x_n1 = 1 + u * (x_n * np.cos(t_n) - y_n * np.sin(t_n))
        y_n1 = u * (x_n * np.sin(t_n) + y_n * np.cos(t_n))

        trajectory[:, n+1, 0] = x_n1
        trajectory[:, n+1, 1] = y_n1

    return trajectory

P = 2000
N = 1000

rng = np.random.default_rng()
x, y = rng.normal(size=(1, P))*10, rng.normal(size=(1, P))*10
u = 0.9

# all_traj = []
# for i in range(P):
#     sol = ikeda_trajectory(u, x[0, i], y[0, i], N)
#     all_traj.append(sol)

# all_traj = np.array(all_traj)
# print(all_traj.shape)
all_traj = ikeda_trajectory(u, x[0, :], y[0, :], N)
print(all_traj.shape)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot()
ax.set_facecolor('#1c1c1c')
ax.set_xticks([])
ax.set_yticks([])
# ax.set_axis_off()
fig.tight_layout()

minx, miny = np.min(np.min(all_traj, axis=1), axis=0)
maxx, maxy = np.max(np.max(all_traj, axis=1), axis=0)

mark, = ax.plot(all_traj[:, 0, 0], all_traj[:, 0, 1], 'o', color='#848484', markersize=1)
# ax.set_xlim(minx, maxx)
# ax.set_ylim(miny, maxy)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)

cache = 10
colors = np.zeros((cache, 4))
colors[:, 3] = np.linspace(0, 0.7, cache, endpoint=True)
scatter = ax.scatter(np.zeros(cache), np.zeros(cache), facecolor=colors, s=1, alpha=0.7)

def update(frame):
    mark.set_data(all_traj[:, frame, 0], all_traj[:, frame, 1])
    # ax.plot(all_traj[:, frame, 0], all_traj[:, frame, 1], 'ko', markersize=1)
    offsets = all_traj[:, max(frame-cache, 0):frame, :].reshape(-1, 2)
    scatter.set_offsets(offsets)
    # ax.autoscale(enable=True, tight=True)

# pbar = tqdm.tqdm(total=N//5)
ani = FuncAnimation(fig, update, interval=50, frames=100)
ani.save('ikeda-black.mp4', writer='ffmpeg')
# ani.save('ikeda-final.gif', writer='imagemagick')
# pbar.close()
# scatter.set_offsets(all_traj[:, 100:140, :])
# scatter.set_offsets(np.array([np.ones(cache), np.ones(cache)]).reshape(-1,2))
# plt.show()
