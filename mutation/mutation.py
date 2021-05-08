import numpy as np
import matplotlib.pyplot as plt
from opensimplex import OpenSimplex
from matplotlib.animation import FuncAnimation
import tqdm

n = 4000
radius = 200
width, height = 500, 500
length = 75
scale = 0.005
two_pi = 2*np.pi
time = 0
rng = np.random.default_rng()
noise = OpenSimplex()

t = rng.uniform(0, 2*np.pi, n)
R = np.sqrt(rng.uniform(0, 1, n))
X, Y = R*np.cos(t), R*np.sin(t)
intensity = np.power(1.5 - np.sqrt(X**2 + Y**2), 2)
X = X*radius + width//2
Y = Y*radius + height//2

def update(frame, scatter, pbar):
    global time
    time += 2*0.002
    cos_t = 1.5*np.cos(two_pi*time)
    sin_t = 1.5*np.sin(two_pi*time)

    offsets = np.zeros((n, 2))
    for i in range(n):
        x, y = X[i], Y[i]
        dx = noise.noise4d(scale*x, scale*y, cos_t, sin_t)*intensity[i]*length
        dy = noise.noise4d(1000+scale*x, 1000+scale*y, cos_t, sin_t)*intensity[i]*length
        offsets[i] = x+dx, y+dy
    pbar.update(1)
    scatter.set_offsets(offsets)
    return scatter

fig, ax = plt.subplots(figsize=(5,5), frameon=False)
# ax.set_facecolor("black")
scatter = ax.scatter(X, Y, s=2.5, edgecolor="none", facecolor="black", alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()

total_frames = 300
pbar = tqdm.tqdm(total=total_frames)
anim = FuncAnimation(fig, update, frames=total_frames, fargs=(scatter, pbar))
pbar2 = tqdm.tqdm(total=total_frames)
anim.save('life-last-white.gif', writer='imagemagick', fps=30, progress_callback=lambda i,n: pbar2.update(1))
pbar.close()
pbar2.close()
