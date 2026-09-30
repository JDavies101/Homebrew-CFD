# airfoil sections: NACA 4-digit coordinates, placement, and an extruded signed distance field
import numpy as np

def naca_four_digit(designation, point_count=200):
    """
    Closed NACA 4-digit section of unit chord (closed trailing edge, cosine spacing), leading edge at (0, 0).

    Returns (x, y): the polygon, upper surface TE -> LE then lower surface LE -> TE, no repeated points.
    """

    max_camber = int(designation[0]) / 100  # m
    camber_position = int(designation[1]) / 10  # p
    thickness = int(designation[2:]) / 100  # t

    # cosine spacing clusters points at both edges
    beta = np.linspace(0.0, np.pi, point_count)
    x = 0.5 * (1.0 - np.cos(beta))
    x_squared = x * x

    # half-thickness, closed trailing edge (last coefficient -0.1036)
    half_thickness = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x_squared + 0.2843 * x_squared * x - 0.1036 * x_squared * x_squared)

    # mean camber line and its slope, piecewise about the max-camber position
    camber = np.zeros_like(x)
    camber_slope = np.zeros_like(x)
    if max_camber > 0:
        front = x < camber_position
        back = ~front

        camber[front] = max_camber / (camber_position * camber_position) * (2 * camber_position * x[front] - x_squared[front])
        camber[back] = max_camber / ((1 - camber_position) * (1 - camber_position)) * ((1 - 2 * camber_position) + 2 * camber_position * x[back] - x_squared[back])
        camber_slope[front] = 2 * max_camber / (camber_position * camber_position) * (camber_position - x[front])
        camber_slope[back] = 2 * max_camber / ((1 - camber_position) * (1 - camber_position)) * (camber_position - x[back])

    # thickness applied normal to the camber line
    slope_angle = np.arctan(camber_slope)
    upper_x = x - half_thickness * np.sin(slope_angle)
    upper_y = camber + half_thickness * np.cos(slope_angle)
    lower_x = x + half_thickness * np.sin(slope_angle)
    lower_y = camber - half_thickness * np.cos(slope_angle)

    # upper TE -> LE, then lower from after the LE to before the TE (the wrap closes the polygon)
    polygon_x = np.concatenate([upper_x[::-1], lower_x[1:-1]])
    polygon_y = np.concatenate([upper_y[::-1], lower_y[1:-1]])

    return polygon_x, polygon_y

def place_section(polygon_x, polygon_y, chord, angle_degrees, leading_edge_x, lowest_y, inverted=True):
    """
    Place a unit-chord section in lattice units: optionally inverted, rotated about the leading edge
    (positive = trailing edge up, more downforce when inverted), scaled, lowest point at lowest_y.

    Returns (x, y) of the placed polygon.
    """

    section_y = -polygon_y if inverted else polygon_y
    angle = np.radians(angle_degrees)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotated_x = polygon_x * cosine - section_y * sine
    rotated_y = polygon_x * sine + section_y * cosine
    placed_x = leading_edge_x + chord * rotated_x
    placed_y = chord * rotated_y

    return placed_x, placed_y + (lowest_y - placed_y.min())

def extruded_section_sdf(polygon_x, polygon_y):
    """
    Signed distance to a closed polygon in (x, y), extruded along z: phi < 0 inside the section.

    Returns a function phi(x, y, z) that works on scalars or whole arrays (z is ignored).
    """

    start_x = np.asarray(polygon_x, np.float64)
    start_y = np.asarray(polygon_y, np.float64)
    end_x = np.roll(start_x, -1)
    end_y = np.roll(start_y, -1)

    def phi(x, y, z):
        shape = np.broadcast(x, y).shape
        points_x = np.broadcast_to(np.asarray(x, np.float64), shape).ravel()
        points_y = np.broadcast_to(np.asarray(y, np.float64), shape).ravel()
        squared_distance = np.full(points_x.shape, np.inf)
        inside = np.zeros(points_x.shape, bool)

        for s in range(len(start_x)):
            segment_x = end_x[s] - start_x[s]
            segment_y = end_y[s] - start_y[s]
            offset_x = points_x - start_x[s]
            offset_y = points_y - start_y[s]

            # closest point on the segment: projection clamped to its ends
            projection = np.clip((offset_x * segment_x + offset_y * segment_y) / (segment_x * segment_x + segment_y * segment_y), 0.0, 1.0)
            gap_x = offset_x - projection * segment_x
            gap_y = offset_y - projection * segment_y
            squared_distance = np.minimum(squared_distance, gap_x * gap_x + gap_y * gap_y)

            # crossing number: a +x ray from the point crosses this segment
            straddles = (start_y[s] > points_y) != (end_y[s] > points_y)
            crossing_x = start_x[s] + offset_y * segment_x / np.where(straddles, segment_y, 1.0)
            inside ^= straddles & (points_x < crossing_x)

        distance = np.sqrt(squared_distance)

        return np.where(inside, -distance, distance).reshape(shape)

    return phi