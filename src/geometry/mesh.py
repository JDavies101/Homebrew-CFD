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
    text = (header + body).decode("ascii", errors="ignore")
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

def inspect_mesh(triangles, cells_per_unit, weld_tolerance=1e-9):
    """
    Size and closure report for an STL triangle soup.

    Returns a dict: lower_corner, upper_corner, size, cells_across (size * cells_per_unit),
    triangle_count, open_edges, nonmanifold_edges, flipped_edges, degenerate_triangles, watertight,
    smallest_radius (STL units, 1st percentile over smoothly curved edges; inf when none).
    """

    # bounding box
    corners = triangles.reshape(-1, 3)
    lower_corner = corners.min(axis=0)
    upper_corner = corners.max(axis=0)
    size = upper_corner - lower_corner

    # weld corners: STL repeats each vertex per triangle
    scale = max(float(size.max()), 1e-30)
    keys = np.round(corners / (scale * weld_tolerance)).astype(np.int64)
    _, vertex_ids = np.unique(keys, axis=0, return_inverse=True)
    vertex_ids = vertex_ids.reshape(-1, 3)

    # degenerate: two corners welded together
    degenerate = (vertex_ids[:, 0] == vertex_ids[:, 1]) | (vertex_ids[:, 1] == vertex_ids[:, 2]) | (vertex_ids[:, 2] == vertex_ids[:, 0])
    
    # directed edges a -> b of every triangle
    edge_start = vertex_ids[:, [0, 1, 2]].ravel()
    edge_end = vertex_ids[:, [1, 2, 0]].ravel()
    undirected = np.sort(np.stack([edge_start, edge_end], axis=1), axis=1)
    _, edge_index, use_count = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)

    # orientation: closed, consistently wound edge is used on a->b and b->a
    forward = (edge_start < edge_end).astype(np.int64)
    forward_count = np.bincount(edge_index.ravel(), weights=forward, minlength=use_count.size)

    open_edges = int(np.sum(use_count == 1))
    nonmanifold_edges = int(np.sum(use_count > 2))
    flipped_edges = int(np.sum((use_count == 2) & (forward_count != 1)))


    # smallest radius of curvature: centroid distance / dihedral angle across edges that bend gently
    # (1 to 60 degrees); sharp creases such as trailing edges and box corners are left out
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    centroids = triangles.mean(axis=1)
    edge_ids = edge_index.ravel()
    order = np.argsort(edge_ids, kind="stable")
    sorted_edges = edge_ids[order]
    owner = np.repeat(np.arange(len(triangles)), 3)[order]
    shared = (sorted_edges[:-1] == sorted_edges[1:]) & (use_count[sorted_edges[:-1]] == 2)
    first = owner[:-1][shared]
    second = owner[1:][shared]
    cosine = np.clip(np.einsum("ij,ij->i", normals[first], normals[second]), -1.0, 1.0)
    angle = np.arccos(cosine)
    distance = np.linalg.norm(centroids[first] - centroids[second], axis=1)
    smooth = (angle > np.radians(1.0)) & (angle < np.radians(60.0))
    radii = distance[smooth] / angle[smooth]
    smallest_radius = float(np.percentile(radii, 1)) if radii.size else float("inf")

    return {
        "lower_corner": lower_corner,
        "upper_corner": upper_corner,
        "size": size,
        "cells_across": size * cells_per_unit,
        "triangle_count": len(triangles),
        "open_edges": open_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "flipped_edges": flipped_edges,
        "degenerate_triangles": int(np.sum(degenerate)),
        "watertight": open_edges == 0 and nonmanifold_edges == 0 and flipped_edges == 0,
        "smallest_radius": smallest_radius
    }

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
