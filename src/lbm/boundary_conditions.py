# bounce-back wall and moving wall
from . import lattice as lt

def bounce_back(populations, solid):
    """
    Reverse every population on the solid nodes (full-way bounce-back).

    Returns the populations after bounce-back.
    """

    bounced_populations = populations.copy()

    for q in range(lt.direction_count):
        bounced_populations[q, solid] = populations[lt.opposite_direction[q], solid]

    return bounced_populations

def moving_wall(populations, lid, wall_velocity_x):
    """
    Bounce back on the lid nodes and add the wall momentum 6 w_q c_qx u_wall.

    Returns the populations after the moving-wall condition.
    """

    bounced_populations = populations.copy()

    for q in range(lt.direction_count):
        bounced_populations[q, lid] = populations[lt.opposite_direction[q], lid] + 6 * lt.lattice_weights[q] * lt.lattice_velocities[q, 0] * wall_velocity_x

    return bounced_populations
