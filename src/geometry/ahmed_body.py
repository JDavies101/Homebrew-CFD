# ahmed body mask: box + rounded nose + rear slant, every size derived from the body height
# body only (no floor): the run adds tunnel walls so drag_body can tell them apart
# staircased: no Bouzidi on the slant or nose yet, stilts omitted
import numpy as np

def ahmed_body(nx, ny, nz, x_start, body_height=48, slant_angle=35, nose="round"):
    """
    Build the Ahmed body solid mask scaled from the 1044 x 288 x 389 mm reference body.

    nose is "round" (100 mm fillet) or "square". slant_angle is in degrees.
    Returns an int32 mask (nx, ny, nz), 1 = solid.
    """

    solid = np.zeros((nx, ny, nz), np.int32)
    # reference body sizes in mm, scaled so 288 mm = body_height cells
    body_length = round(1044 / 288 * body_height)
    body_width = round(389 / 288 * body_height)
    nose_radius = round(100 / 288 * body_height)
    ground_clearance = round(50 / 288 * body_height)
    slant_length = round(222 / 288 * body_height)  # slant surface length (fixed)
    slant_length_x = round(slant_length * np.cos(np.radians(slant_angle)))  # horizontal projection
    slant_drop_y = round(slant_length * np.sin(np.radians(slant_angle)))  # vertical drop

    # body floats above the clearance gap
    j_bottom = ground_clearance + 1
    j_top = j_bottom + body_height
    x_rear = x_start + body_length
    # centered spanwise
    k_start = (nz - body_width) // 2
    k_end = k_start + body_width

    # main box ahead of the slant
    solid[x_start : x_rear - slant_length_x, j_bottom : j_top, k_start : k_end] = 1

    # rear slant
    for i in range(x_rear - slant_length_x, x_rear):
        slant_top_j = j_top - round(slant_drop_y * (i - (x_rear - slant_length_x)) / slant_length_x)
        solid[i, j_bottom : slant_top_j, k_start : k_end] = 1

    # rounded nose (square nose: keep the sharp box front)
    if nose == "round":
        i = np.arange(x_start, x_start + nose_radius)[:, None, None]
        j = np.arange(j_bottom, j_top)[None, :, None]
        k = np.arange(k_start, k_end)[None, None, :]
        edge_distance_y = np.minimum(j - j_bottom, (j_top - 1) - j)  # distance to nearest top/bottom edge
        edge_distance_z = np.minimum(k - k_start, (k_end - 1) - k)  # distance to nearest side edge

        # how far the fillet pushes the front face back, per axis (0 when that edge is far)
        gap_y = nose_radius - edge_distance_y
        gap_z = nose_radius - edge_distance_z
        # np.maximum(..., 0) keeps sqrt off negatives on the discarded branch of np.where
        setback_y = np.where(edge_distance_y < nose_radius,
                             nose_radius - np.sqrt(np.maximum(nose_radius * nose_radius - gap_y * gap_y, 0)), 0)
        setback_z = np.where(edge_distance_z < nose_radius,
                             nose_radius - np.sqrt(np.maximum(nose_radius * nose_radius - gap_z * gap_z, 0)), 0)

        carve = i < x_start + np.maximum(setback_y, setback_z)  # broadcasts to (nose_radius, H, W)
        solid[x_start : x_start + nose_radius, j_bottom : j_top, k_start : k_end][carve] = 0

    return solid
