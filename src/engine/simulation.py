# 2D D2Q9 taichi solver, runs on cpu or cuda
import taichi as ti
import numpy as np
from src.engine import lattice_d2q9 as d2q9
from src.engine import runtime

@ti.data_oriented
class Simulation:
    """
    2D D2Q9 BGK solver on Taichi fields: collide, stream, bounce-back, moving lid.
    """

    def __init__(self, nx, ny, backend="cpu"):
        """
        Allocate populations, macroscopic fields, masks and lattice constants.
        """

        runtime.init(backend)
        self.direction_count = d2q9.direction_count
        self.dimension = d2q9.dimension
        self.nx, self.ny = nx, ny

        # populations + macroscopic (f, rho, u kept short: shared engine API)
        self.f = ti.field(ti.f32, shape=(self.direction_count, nx, ny))
        self.f_new = ti.field(ti.f32, shape=(self.direction_count, nx, ny))
        self.rho = ti.field(ti.f32, shape=(nx, ny))
        self.u = ti.field(ti.f32, shape=(self.dimension, nx, ny))
        self.solid = ti.field(ti.i32, shape=(nx, ny))
        self.lid = ti.field(ti.i32, shape=(nx, ny))

        # lattice constants as fields, built from the NumPy descriptor
        self.lattice_velocities = ti.field(ti.i32, shape=(self.direction_count, self.dimension))
        self.lattice_velocities.from_numpy(d2q9.lattice_velocities.astype(np.int32))
        self.lattice_weights = ti.field(ti.f32, shape=self.direction_count)
        self.lattice_weights.from_numpy(d2q9.lattice_weights.astype(np.float32))
        self.opposite_direction = ti.field(ti.i32, shape=self.direction_count)
        self.opposite_direction.from_numpy(d2q9.opposite_direction.astype(np.int32))

    @ti.kernel
    def macroscopic(self):
        """
        Density and velocity from the populations, parallel over cells.
        """

        for i, j in ti.ndrange(self.nx, self.ny):
            density = 0.0
            momentum_x = 0.0
            momentum_y = 0.0

            # serial: sum the 9 populations
            for q in range(self.direction_count):
                density += self.f[q, i, j]
                momentum_x += self.f[q, i, j] * self.lattice_velocities[q, 0]
                momentum_y += self.f[q, i, j] * self.lattice_velocities[q, 1]

            self.rho[i, j] = density
            self.u[0, i, j] = momentum_x / density
            self.u[1, i, j] = momentum_y / density

    @ti.kernel
    def collide(self, relaxation_time: ti.f32):
        """
        BGK relaxation toward equilibrium on fluid cells, parallel over cells.
        """

        for i, j in ti.ndrange(self.nx, self.ny):
            # wall nodes only hold bounced populations; don't collide them
            if self.solid[i, j] == 1 or self.lid[i, j] == 1:
                continue

            # moments
            density = 0.0
            momentum_x = 0.0
            momentum_y = 0.0
            for q in range(self.direction_count):
                density += self.f[q, i, j]
                momentum_x += self.f[q, i, j] * self.lattice_velocities[q, 0]
                momentum_y += self.f[q, i, j] * self.lattice_velocities[q, 1]
            velocity_x = momentum_x / density
            velocity_y = momentum_y / density
            velocity_squared = velocity_x * velocity_x + velocity_y * velocity_y

            # equilibrium + BGK relax, per direction
            for q in range(self.direction_count):
                velocity_dot_direction = self.lattice_velocities[q, 0] * velocity_x + self.lattice_velocities[q, 1] * velocity_y
                equilibrium_population = self.lattice_weights[q] * density * (1 + 3 * velocity_dot_direction + 4.5 * velocity_dot_direction * velocity_dot_direction - 1.5 * velocity_squared)
                self.f[q, i, j] += -(1 / relaxation_time) * (self.f[q, i, j] - equilibrium_population)

    @ti.kernel
    def _stream(self):
        """
        Pull streaming into f_new with periodic wrap, parallel over destination cells.
        """

        for i, j in ti.ndrange(self.nx, self.ny):
            for q in range(self.direction_count):
                # where this population came from; % nx / ny = periodic wrap
                source_i = (i - self.lattice_velocities[q, 0]) % self.nx
                source_j = (j - self.lattice_velocities[q, 1]) % self.ny
                self.f_new[q, i, j] = self.f[q, source_i, source_j]

    def stream(self):
        """
        Stream kernel, then copy the buffer back (plain Python, not a kernel).
        """

        self._stream()
        self.f.copy_from(self.f_new)

    @ti.kernel
    def bounce_back(self):
        """
        Swap opposite populations on solid nodes.
        """

        for i, j in ti.ndrange(self.nx, self.ny):
            if self.solid[i, j] == 1:
                for q in range(self.direction_count):
                    opposite = self.opposite_direction[q]
                    # visit each opposite pair once
                    if q < opposite:
                        swap = self.f[q, i, j]
                        self.f[q, i, j] = self.f[opposite, i, j]
                        self.f[opposite, i, j] = swap

    @ti.kernel
    def moving_wall(self, wall_velocity_x: ti.f32):
        """
        Bounce-back on lid nodes plus the wall momentum 6 w_q c_qx u_wall.
        """

        for i, j in ti.ndrange(self.nx, self.ny):
            if self.lid[i, j] == 1:
                for q in range(self.direction_count):
                    opposite = self.opposite_direction[q]
                    if q < opposite:
                        # bounce-back
                        swap = self.f[q, i, j]
                        self.f[q, i, j] = self.f[opposite, i, j]
                        self.f[opposite, i, j] = swap
                        # wall momentum, the opposite direction gets the negative
                        correction = 6.0 * self.lattice_weights[q] * self.lattice_velocities[q, 0] * wall_velocity_x
                        self.f[q, i, j] += correction
                        self.f[opposite, i, j] -= correction

    def step(self, relaxation_time, wall_velocity_x=0.0):
        """
        One timestep: collide, stream, bounce-back, moving lid (plain Python, not a kernel).
        """

        self.collide(relaxation_time)
        self.stream()
        self.bounce_back()
        self.moving_wall(wall_velocity_x)

    def run(self, steps, relaxation_time, wall_velocity_x=0.0):
        """
        Advance the given number of steps.
        """

        for _ in range(steps):
            self.step(relaxation_time, wall_velocity_x)
