# 2D timestep and run loop, with Guo body force
import numpy as np
from .stream import stream
from .boundary_conditions import bounce_back
from .equilibrium import equilibrium
from . import lattice as lt
from .moments import macroscopic

def guo_source(velocity, force, relaxation_time):
    """
    Compute the Guo forcing term for every direction and cell.

    Returns the source term (9, nx, ny).
    """

    velocity_dot_direction = np.einsum("qc,cxy->qxy", lt.lattice_velocities, velocity)
    force_dot_direction = np.einsum("qc,cxy->qxy", lt.lattice_velocities, force)
    velocity_dot_force = np.einsum("cxy,cxy->xy", velocity, force)

    prefactor = 1 - 1 / (2 * relaxation_time)
    source = prefactor * lt.lattice_weights[:, None, None] * (3 * (force_dot_direction - velocity_dot_force) + 9 * velocity_dot_direction * force_dot_direction)

    return source

def collide_forced(populations, relaxation_time, force, solid=None):
    """
    BGK collision with the Guo force: half-force velocity shift plus source term.

    Returns the post-collision populations.
    """

    density, raw_velocity = macroscopic(populations)
    velocity = raw_velocity + force / (2 * density)
    equilibrium_populations = equilibrium(density, velocity)
    source = guo_source(velocity, force, relaxation_time)

    collided_populations = populations - (1 / relaxation_time) * (populations - equilibrium_populations) + source
    if solid is not None:
        collided_populations = np.where(solid[None], populations, collided_populations)  # no relaxation, no force at walls

    return collided_populations

def step(populations, relaxation_time, solid, body_force_x=0):
    """
    Advance one timestep: forced collision, streaming, bounce-back.

    Returns the populations after one step.
    """

    nx, ny = populations.shape[1], populations.shape[2]
    force = np.zeros((2, nx, ny))
    force[0] = body_force_x
    collided_populations = collide_forced(populations, relaxation_time, force, solid)
    streamed_populations = stream(collided_populations)
    bounced_populations = bounce_back(streamed_populations, solid)

    return bounced_populations

def run(populations, relaxation_time, solid, steps, body_force_x=0):
    """
    Advance the populations a given number of steps.

    Returns the final populations.
    """

    for _ in range(steps):
        populations = step(populations, relaxation_time, solid, body_force_x)

    return populations

def initial(nx, ny):
    """
    Build equilibrium populations at unit density and rest.

    Returns the initial populations (9, nx, ny).
    """

    density = np.ones((nx, ny))
    velocity = np.zeros((2, nx, ny))
    populations = equilibrium(density, velocity)

    return populations
