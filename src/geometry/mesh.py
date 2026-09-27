# triangle meshes: STL read/write, lattice placement, and a test mesh (icosphere)
import numpy as np

def read_stl(path):
    # returns float64 array (T, 3, 3): T triangles x 3 corners x (x, y, z)
    with open(path, "rb") as fh:
        head = fh.read(80)
        count = np.frombuffer(fh.read(4), np.uint32)
        body = fh.read()

    if count.size == 1 and len(body) == int(count[0]) * 50:
        rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3,3)), ("attr", "<u2")])
        return np.frombuffer(body, rec)["v"].astype(np.float64)
    
    text = (head + body).decode("ascii", erros="ignore")
    verts = [line.split()[1:4] for line in text.splitlines() if line.strip().startswith("vertex")]
    return np.array(verts, np.float64).reshape(-1, 3, 3)

def write_stl(path, tris):
    # binary STL; normals written as zero (readers recompute them)
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    out = np.zeros(len(tris), rec)
    out["v"] = tris
    with open(path, "wb") as fh:
        fh.write(b"homebrew-cfd".ljust(80, b" "))
        fh.write(np.uint32(len(tris)).tobytes())
        fh.write(out.tobytes())

def to_lattice(tris, origin, dx):
    # physical coordinates -> lattice units: node (i, j, k) sits at origin + (i, j, k) * dx
    return (tris - np.asarray(origin, np.float64)) / dx

def icosphere(center, radius, subdiv=3):
    # closed triangle mesh of a sphere: icosahedron, each face split into 4, points pushed to the sphere
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    verts = [np.array(p, np.float64) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(subdiv):
        cache = {}
        new_faces = []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = verts[i] + verts[j]
                verts.append(m / np. linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        
        for a, b, c in faces:
            ab = mid(a, b)
            bc = mid(b, c)
            ca = mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        
        faces = new_faces

    V = np.array(verts)
    F = np.array(faces)
    return np.asarray(center, np.float64) + radius * V[F]