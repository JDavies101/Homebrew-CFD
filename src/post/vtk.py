# export 3D fields to a .vti file for ParaView
import os
import numpy as np
from pyevtk.hl import imageToVTK

# rho: (nx,ny,nz)   u: (3,nx,ny,nz)   -> writes <path>.vti
def write_field(path, rho, u):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)   # pyevtk won't create the folder
    ux = np.ascontiguousarray(u[0])
    uy = np.ascontiguousarray(u[1])
    uz = np.ascontiguousarray(u[2])
    speed = np.sqrt(ux**2 + uy**2 + uz**2)
    imageToVTK(path, pointData={
        "rho": np.ascontiguousarray(rho),
        "u": (ux, uy, uz),          # a proper vector field in ParaView
        "speed": speed,
    })
    print(f"wrote {os.path.abspath(path)}.vti")
