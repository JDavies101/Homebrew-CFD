# export 3D fields to a .vti file for ParaView
import os
import numpy as np
from pyevtk.hl import imageToVTK

def write_field(path, density, velocity):
    """
    Write density (nx, ny, nz), velocity (3, nx, ny, nz) and speed to <path>.vti.
    """

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)  # pyevtk won't create the folder

    velocity_x = np.ascontiguousarray(velocity[0])
    velocity_y = np.ascontiguousarray(velocity[1])
    velocity_z = np.ascontiguousarray(velocity[2])
    speed = np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z)
    imageToVTK(path, pointData={
        "rho": np.ascontiguousarray(density),
        "u": (velocity_x, velocity_y, velocity_z),  # a proper vector field in ParaView
        "speed": speed,
    })

    print(f"wrote {os.path.abspath(path)}.vti")
