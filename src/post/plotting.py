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

# channel law of the wall: mean u+(y+) over the sublayer + log law, and the rms profiles.
# all inputs already in wall units (y+, u+, rms/u_tau), one wall-normal half.
def plot_law_of_wall(y_plus, u_plus, urms, vrms, wrms):
    yp_ref = np.logspace(np.log10(y_plus.min()), np.log10(y_plus.max()), 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.semilogx(y_plus, u_plus, 'o-', ms=3, label='LBM')
    ax1.semilogx(yp_ref, yp_ref, 'k:', label='u+ = y+')
    ax1.semilogx(yp_ref, np.log(yp_ref) / 0.41 + 5.2, 'k--', label='log law')  # kappa=0.41 B=5.2
    ax1.set(xlabel='y+', ylabel='u+', ylim=(0, 20), title='mean velocity'); ax1.legend()
    ax2.semilogx(y_plus, urms, label="u'")
    ax2.semilogx(y_plus, vrms, label="v'")
    ax2.semilogx(y_plus, wrms, label="w'")
    ax2.set(xlabel='y+', ylabel='rms / u_tau', title='fluctuations'); ax2.legend()
    fig.tight_layout()
    return fig, (ax1, ax2)