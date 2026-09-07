# 2D field plots: velocity magnitude and streamlines
import numpy as np
import matplotlib.pyplot as plt

# heatmap of |u|. .T + origin lower so x is horizontal and y points up
def plot_velocity_magnitude(u):
    speed = np.sqrt(u[0]**2 + u[1]**2)
    fig, ax = plt.subplots()
    im = ax.imshow(speed.T, origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label='|u|')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    return fig, ax

# streamlines coloured by speed
def plot_streamlines(u):
    nx, ny = u.shape[1], u.shape[2]
    x, y = np.arange(nx), np.arange(ny)
    fig, ax = plt.subplots()
    ax.streamplot(x, y, u[0].T, u[1].T, color=np.sqrt(u[0]**2+u[1]**2).T, cmap='viridis', density=1.5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    return fig, ax