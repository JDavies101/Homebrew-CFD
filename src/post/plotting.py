# plots: 2D fields (speed, streamlines), channel law of the wall, 3D mask / velocity slices
import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_magnitude(velocity):
    """
    Heatmap of 2D speed; .T + origin lower so x is horizontal and y points up.

    Returns fig, ax.
    """

    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    fig, ax = plt.subplots()
    image = ax.imshow(speed.T, origin="lower", cmap="viridis")
    fig.colorbar(image, ax=ax, label="|u|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax

def plot_streamlines(velocity):
    """
    2D streamlines coloured by speed.

    Returns fig, ax.
    """

    nx, ny = velocity.shape[1], velocity.shape[2]
    x, y = np.arange(nx), np.arange(ny)
    fig, ax = plt.subplots()
    ax.streamplot(x, y, velocity[0].T, velocity[1].T, color=np.sqrt(velocity[0] ** 2 + velocity[1] ** 2).T, cmap="viridis", density=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return fig, ax

def plot_law_of_wall(y_plus, u_plus, u_rms, v_rms, w_rms):
    """
    Channel law of the wall: mean u+(y+) against the sublayer and log law, and the rms profiles.
    All inputs already in wall units (y+, u+, rms / u_tau), one wall-normal half.

    Returns fig, (ax_mean, ax_rms).
    """

    y_plus_reference = np.logspace(np.log10(y_plus.min()), np.log10(y_plus.max()), 200)
    fig, (ax_mean, ax_rms) = plt.subplots(1, 2, figsize=(11, 4))

    ax_mean.semilogx(y_plus, u_plus, "o-", ms=3, label="LBM")
    ax_mean.semilogx(y_plus_reference, y_plus_reference, "k:", label="u+ = y+")
    ax_mean.semilogx(y_plus_reference, np.log(y_plus_reference) / 0.41 + 5.2, "k--", label="log law")  # kappa 0.41, B 5.2
    ax_mean.set(xlabel="y+", ylabel="u+", ylim=(0, 20), title="mean velocity")
    ax_mean.legend()

    ax_rms.semilogx(y_plus, u_rms, label="u'")
    ax_rms.semilogx(y_plus, v_rms, label="v'")
    ax_rms.semilogx(y_plus, w_rms, label="w'")
    ax_rms.set(xlabel="y+", ylabel="rms / u_tau", title="fluctuations")
    ax_rms.legend()

    fig.tight_layout()

    return fig, (ax_mean, ax_rms)

def plot_mask_slice(solid, axis, index):
    """
    2D slice of a solid mask for geometry checks; the two remaining axes become horizontal, vertical.

    Returns fig, ax.
    """

    axis_names = ["x", "y", "z"]
    mask_slice = np.take(solid, index, axis=axis)  # 2D: the two axes that remain
    horizontal, vertical = [name for i, name in enumerate(axis_names) if i != axis]
    fig, ax = plt.subplots()
    ax.imshow(mask_slice.T, origin="lower", cmap="gray_r", interpolation="nearest", aspect="equal")
    ax.set_xlabel(horizontal)
    ax.set_ylabel(vertical)

    return fig, ax

def plot_velocity_slice(velocity, axis, index, component=0):
    """
    2D slice of one velocity component (0 = x, 1 = y, 2 = z), diverging about 0 so backflow reads blue.

    Returns fig, ax.
    """

    axis_names = ["x", "y", "z"]
    component_slice = np.take(velocity[component], index, axis=axis)  # 2D component field
    horizontal, vertical = [name for i, name in enumerate(axis_names) if i != axis]
    color_limit = float(np.nanmax(np.abs(component_slice))) or 1.0
    fig, ax = plt.subplots()
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("0.45")  # NaN = solid, grey
    image = ax.imshow(component_slice.T, origin="lower", cmap=colormap, vmin=-color_limit, vmax=color_limit,
                      interpolation="nearest", aspect="equal")
    fig.colorbar(image, ax=ax, label=f"u_{axis_names[component]}")
    ax.set_xlabel(horizontal)
    ax.set_ylabel(vertical)

    return fig, ax
