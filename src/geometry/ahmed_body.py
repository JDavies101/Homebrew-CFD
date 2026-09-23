# ahmed body mask: box + rounded nose + rear slant at phi, every size derived from height H
# body only (no floor): the run adds tunnel walls so drag_body can tell them apart
# staircased: no Bouzidi on the slant or nose yet, stilts omitted
import numpy as np

def ahmed_body(nx, ny, nz, x0, H=48, phi=35):
    solid = np.zeros((nx, ny, nz), np.int32)
    Hb = H # cells
    Lb = round(1044 / 288 * H)
    Wb = round(389 / 288 * H)
    R = round(100/288 * H)
    clear = round(50 / 288 * H)
    slant_len = round(222 / 288 * H)                        # slant surface length (fixed)
    slant_dx  = round(slant_len * np.cos(np.radians(phi)))  # horizontal projection
    slant_dy  = round(slant_len * np.sin(np.radians(phi)))  # vertical drop    
    # body floats above the clearance gap
    j_bot = clear + 1

    # top of body
    j_top = j_bot + Hb

    x_rear = x0 + Lb
    # centered spanwise
    k0 = (nz - Wb) // 2
    k1 = k0 + Wb

    # main box ahead of slant
    solid[x0 : x_rear - slant_dx, j_bot : j_top, k0 : k1] = 1

    # rear slant
    for i in range(x_rear - slant_dx, x_rear):
        top_j = j_top - round(slant_dy * (i - (x_rear - slant_dx)) / slant_dx)
        solid[i, j_bot : top_j, k0 : k1] = 1

    # rounding nose
    for i in range(x0, x0 + R):
        for j in range(j_bot, j_top):
            for k in range(k0, k1):
                dy = min(j - j_bot, (j_top - 1) - j)  # distance to nearest top/bottom edge
                dz = min(k - k0, (k1 - 1) - k) # distance to nearest side edge
                # how far the fillet pushes the front face back, per axis (0 when that edge is far)
                back_y = R - np.sqrt(R ** 2 - (R - dy) ** 2) if dy < R else 0
                back_z = R - np.sqrt(R ** 2 - (R - dz) ** 2) if dz < R else 0

                if i < x0 + max(back_y, back_z):
                    solid[i, j, k] = 0

    return solid