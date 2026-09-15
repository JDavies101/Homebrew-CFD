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

# 2D slice of a solid mask for geometry checks. axis = the axis to slice through;
# the two remaining axes become horizontal, vertical (origin lower, so up is up).
def plot_mask_slice(solid, axis, index):
    names = ['x', 'y', 'z']
    sl = np.take(solid, index, axis=axis)          # 2D: the two axes that remain
    h, v = [n for i, n in enumerate(names) if i != axis]
    fig, ax = plt.subplots()
    ax.imshow(sl.T, origin='lower', cmap='gray_r', interpolation='nearest', aspect='equal')
    ax.set_xlabel(h); ax.set_ylabel(v)
    return fig, ax

# 2D slice of a velocity component (comp: 0=x,1=y,2=z), diverging about 0 so
# backflow (negative) reads blue -> shows separation/recirculation at a glance.
def plot_velocity_slice(u, axis, index, comp=0):
    names = ['x', 'y', 'z']
    sl = np.take(u[comp], index, axis=axis)        # 2D component field
    h, v = [n for i, n in enumerate(names) if i != axis]
    lim = float(np.nanmax(np.abs(sl))) or 1.0
    fig, ax = plt.subplots()
    im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                   interpolation='nearest', aspect='equal')
    fig.colorbar(im, ax=ax, label=f'u_{names[comp]}')
    ax.set_xlabel(h); ax.set_ylabel(v)
    return fig, ax