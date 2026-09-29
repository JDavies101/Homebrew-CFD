# triangle meshes: STL read/write, lattice placement, and test meshes (icosphere, box)
import numpy as np

def read_stl(path):
    """
    Read a binary or ASCII STL file.

    Returns float64 triangles (T, 3, 3): T triangles x 3 corners x (x, y, z).
    """

    with open(path, "rb") as file:
        header = file.read(80)
        triangle_count = np.frombuffer(file.read(4), np.uint32)
        body = file.read()

    # binary STL: 50 bytes per triangle record
    if triangle_count.size == 1 and len(body) == int(triangle_count[0]) * 50:
        record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        return np.frombuffer(body, record)["v"].astype(np.float64)

    # ASCII STL: collect every vertex line
    text = (header + body).decode("ascii", erros="ignore")
    vertices = [line.split()[1:4] for line in text.splitlines() if line.strip().startswith("vertex")]

    return np.array(vertices, np.float64).reshape(-1, 3, 3)

def write_stl(path, triangles):
    """
    Write triangles (T, 3, 3) as a binary STL; normals written as zero (readers recompute them).
    """

    record = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    records = np.zeros(len(triangles), record)
    records["v"] = triangles

    with open(path, "wb") as file:
        file.write(b"homebrew-cfd".ljust(80, b" "))
        file.write(np.uint32(len(triangles)).tobytes())
        file.write(records.tobytes())

def to_lattice(triangles, origin, spacing):
    """
    Physical coordinates to lattice units: node (i, j, k) sits at origin + (i, j, k) * spacing.

    Returns triangles (T, 3, 3) in lattice units.
    """

    return (triangles - np.asarray(origin, np.float64)) / spacing

def icosphere(center, radius, subdivisions=3):
    """
    Closed triangle mesh of a sphere: icosahedron, each face split into 4, points pushed to the sphere.

    Returns triangles (T, 3, 3).
    """

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    icosahedron_points = [[-1, golden_ratio, 0], [1, golden_ratio, 0], [-1, -golden_ratio, 0], [1, -golden_ratio, 0],
                          [0, -1, golden_ratio], [0, 1, golden_ratio], [0, -1, -golden_ratio], [0, 1, -golden_ratio],
                          [golden_ratio, 0, -1], [golden_ratio, 0, 1], [-golden_ratio, 0, -1], [-golden_ratio, 0, 1]]
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
             [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
             [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    vertices = [np.array(point, np.float64) / np.linalg.norm(point) for point in icosahedron_points]

    for _ in range(subdivisions):
        midpoint_cache = {}
        new_faces = []

        def midpoint(i, j):
            # index of the (cached) edge midpoint, pushed out to the unit sphere
            key = (min(i, j), max(i, j))
            if key not in midpoint_cache:
                edge_sum = vertices[i] + vertices[j]
                vertices.append(edge_sum / np.linalg.norm(edge_sum))
                midpoint_cache[key] = len(vertices) - 1

            return midpoint_cache[key]

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

        faces = new_faces

    vertex_array = np.array(vertices)
    face_array = np.array(faces)

    return np.asarray(center, np.float64) + radius * vertex_array[face_array]

def box_mesh(lower_corner, upper_corner):
    """
    Closed axis-aligned box.

    Returns 12 triangles (12, 3, 3).
    """

    x0, y0, z0 = lower_corner
    x1, y1, z1 = upper_corner

    vertices = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                         [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])

    return vertices[faces]
