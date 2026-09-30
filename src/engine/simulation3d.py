# 3D D3Q19 LBM solver, runs on cpu or cuda
# geometry-agnostic: consumes solid/body/lid/q fields, does not build shapes
import taichi as ti
import numpy as np
from src.engine import lattice_d3q19 as d3q19
from src.engine import runtime
from src.geometry.sdf import normals_from_sdf

@ti.data_oriented
class Simulation3D:
    """
    3D D3Q19 lattice Boltzmann solver on Taichi fields.

    Public fields keep short LBM names shared across the engine, examples and run log:
    f (populations), rho (density), u (velocity), uw (wall velocity), solid, body, lid.
    """

    def __init__(self, nx, ny, nz, backend="cpu", interp=False, max_parts=8):
        """
        Allocate fields and load the lattice constants. interp=True adds the Bouzidi fields (q, fc).
        max_parts is the largest part id body may hold (part_force has max_parts + 1 rows).
        """

        runtime.init(backend)
        self.direction_count = d3q19.direction_count
        self.dimension = d3q19.dimension
        self.nx, self.ny, self.nz = nx, ny, nz

        # populations + macroscopic
        self.f = ti.field(ti.f32, shape=(self.direction_count, nx, ny, nz))
        self.f_new = ti.field(ti.f32, shape=(self.direction_count, nx, ny, nz))
        self.rho = ti.field(ti.f32, shape=(nx, ny, nz))
        self.u = ti.field(ti.f32, shape=(self.dimension, nx, ny, nz))
        self.solid = ti.field(ti.i32, shape=(nx, ny, nz))
        self.lid = ti.field(ti.i32, shape=(nx, ny, nz))
        self.uw = ti.field(ti.f32, shape=(self.dimension, nx, ny, nz))  # wall velocity at solid nodes (0 = static)
        self.force = ti.field(ti.f32, shape=self.dimension)

        # Bouzidi only: wall fractions + post-collision snapshot
        if interp:
            self.q = ti.field(ti.f32, shape=(self.direction_count, nx, ny, nz))  # wall fractions (0 = not a boundary link)
            self.fc = ti.field(ti.f32, shape=(self.direction_count, nx, ny, nz))  # post-collision snapshot (Bouzidi needs it)

        # eddy viscosities, relaxation layers, drag mask
        self.nut_wall = ti.field(ti.f32, shape=(nx, ny, nz))  # wall-model eddy viscosity (0 away from walls)
        self.nut_les = ti.field(ti.f32, shape=(nx, ny, nz))  # WALE subgrid eddy viscosity (0 until les_wale runs)
        self.sigma = ti.field(ti.f32, shape=(nx, nz))  # relaxation layer strength per (x, z) column, 0 by default
        self.sigma_z = ti.field(ti.f32, shape=(nx, nz))  # z-layer strength (relaxes to running mean), 0 by default
        self.rho_bar = ti.field(ti.f32, shape=(nx, ny, nz))  # running-mean density, z-layer target
        self.u_bar = ti.field(ti.f32, shape=(self.dimension, nx, ny, nz))  # running-mean velocity, z-layer target
        self.body = ti.field(ti.i32, shape=(nx, ny, nz))  # part id per solid node for drag: 0 = not measured (tunnel walls), 1..max_parts = part
        self.max_parts = max_parts
        self.part_force = ti.field(ti.f32, shape=(max_parts + 1, self.dimension))  # force per part id (row 0 unused)

        # lattice constants as fields, built from the NumPy descriptor
        self.lattice_velocities = ti.field(ti.i32, shape=(self.direction_count, self.dimension))  # c_q (E)
        self.lattice_velocities.from_numpy(d3q19.lattice_velocities.astype(np.int32))
        self.lattice_weights = ti.field(ti.f32, shape=self.direction_count)  # w_q (W)
        self.lattice_weights.from_numpy(d3q19.lattice_weights.astype(np.float32))
        self.opposite_direction = ti.field(ti.i32, shape=self.direction_count)  # OPP
        self.opposite_direction.from_numpy(d3q19.opposite_direction.astype(np.int32))
        self.mirror_y = ti.field(ti.i32, shape=self.direction_count)  # MIRROR_Y
        self.mirror_y.from_numpy(d3q19.mirror_y.astype(np.int32))
        self.mirror_z = ti.field(ti.i32, shape=self.direction_count)  # MIRROR_Z
        self.mirror_z.from_numpy(d3q19.mirror_z.astype(np.int32))

    @ti.kernel
    def macroscopic(self):
        """
        Density and velocity from the populations, parallel over cells.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            moments = self._moments(i, j, k, 0.0)
            self.rho[i, j, k] = moments[0]
            self.u[0, i, j, k] = moments[1]
            self.u[1, i, j, k] = moments[2]
            self.u[2, i, j, k] = moments[3]

    # wrappers: each of these is a special case of collide_full (collide_reg is separate)

    def collide(self, relaxation_time: ti.f32):
        """
        Plain BGK.
        """

        self.collide_full(relaxation_time, 0.0, 0.0, 0)

    def collide_forced(self, relaxation_time: ti.f32, body_force_x: ti.f32):
        """
        BGK + Guo body force in x.
        """

        self.collide_full(relaxation_time, 0.0, body_force_x, 0)

    def collide_trt(self, relaxation_time: ti.f32):
        """
        Two-relaxation-time.
        """

        self.collide_full(relaxation_time, 0.0, 0.0, 1)

    def collide_les(self, relaxation_time: ti.f32, smagorinsky_constant: ti.f32):
        """
        BGK + Smagorinsky subgrid viscosity.
        """

        self.collide_full(relaxation_time, smagorinsky_constant, 0.0, 0)

    @ti.kernel
    def _collide_reg(self, base_relaxation_time: ti.f32, smagorinsky_constant: ti.f32, body_force_x: ti.f32,
                     inlet_velocity: ti.f32, mean_update_rate: ti.f32, z_layers_on: ti.i32):
        """
        Regularized collision: rebuild f_neq from the stress Pi only (drops the ghost moments that blow up
        as tau -> 0.5), then relax. Single rate, no TRT split. smagorinsky_constant = 0 disables LES,
        body_force_x = 0 disables forcing. Forcing is supported: the first-order Hermite term restores the
        -F/2 first moment that regularization would otherwise drop. Relaxation layers are fused in.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):

            # wall nodes only hold bounced populations for one step, don't collide them
            if self.solid[i, j, k] == 1 or self.lid[i, j, k] == 1:
                continue

            # moments
            moments = self._moments(i, j, k, body_force_x)
            density = moments[0]
            velocity_x = moments[1]
            velocity_y = moments[2]
            velocity_z = moments[3]
            velocity_squared = moments[4]

            # pass 1: non-equilibrium stress tensor Pi = sum c_a c_b (f - feq)
            stress = self._stress(i, j, k, density, velocity_x, velocity_y, velocity_z, velocity_squared)
            stress_xx = stress[0]
            stress_yy = stress[1]
            stress_zz = stress[2]
            stress_xy = stress[3]
            stress_xz = stress[4]
            stress_yz = stress[5]
            trace = stress_xx + stress_yy + stress_zz

            # pass 2: reconstruct f_neq from Pi only, then relax
            relaxation_time = base_relaxation_time
            # LES: eddy-adjusted tau from the stress magnitude
            if smagorinsky_constant > 0.0:
                stress_magnitude = ti.sqrt(stress_xx * stress_xx + stress_yy * stress_yy + stress_zz * stress_zz + 2.0 * (stress_xy * stress_xy + stress_xz * stress_xz + stress_yz * stress_yz))
                relaxation_time = 0.5 * (base_relaxation_time + ti.sqrt(base_relaxation_time * base_relaxation_time + 18.0 * ti.sqrt(2.0) * smagorinsky_constant * smagorinsky_constant * stress_magnitude / density))
            relaxation_rate = 1.0 / (relaxation_time + 3.0 * (self.nut_wall[i, j, k] + self.nut_les[i, j, k]))  # omega
            guo_prefactor = 1.0 - 0.5 * relaxation_rate  # BGK single rate

            # absorbing layers, fused: strengths for this column
            layer_strength_x = self.sigma[i, k]
            layer_strength_z = 0.0
            if z_layers_on == 1:
                layer_strength_z = self.sigma_z[i, k]

            # moments after collision (Guo: j = rho u + F/2) and after the x layer, in closed form,
            # because relaxing toward feq(1, inlet_velocity) moves (rho, j) toward (1, inlet_velocity) by the same fraction
            momentum_x = density * velocity_x + 0.5 * body_force_x
            momentum_y = density * velocity_y
            momentum_z = density * velocity_z
            layer_density = density + layer_strength_x * (1.0 - density)
            layer_momentum_x = momentum_x + layer_strength_x * (inlet_velocity - momentum_x)
            layer_momentum_y = momentum_y * (1.0 - layer_strength_x)
            layer_momentum_z = momentum_z * (1.0 - layer_strength_x)

            # running-mean target for the z layers
            mean_density = 1.0
            mean_velocity_x = 0.0
            mean_velocity_y = 0.0
            mean_velocity_z = 0.0
            if layer_strength_z > 0.0:
                self.rho_bar[i, j, k] += mean_update_rate * (layer_density - self.rho_bar[i, j, k])
                self.u_bar[0, i, j, k] += mean_update_rate * (layer_momentum_x / layer_density - self.u_bar[0, i, j, k])
                self.u_bar[1, i, j, k] += mean_update_rate * (layer_momentum_y / layer_density - self.u_bar[1, i, j, k])
                self.u_bar[2, i, j, k] += mean_update_rate * (layer_momentum_z / layer_density - self.u_bar[2, i, j, k])
                mean_density = self.rho_bar[i, j, k]
                mean_velocity_x = self.u_bar[0, i, j, k]
                mean_velocity_y = self.u_bar[1, i, j, k]
                mean_velocity_z = self.u_bar[2, i, j, k]
            mean_velocity_squared = mean_velocity_x * mean_velocity_x + mean_velocity_y * mean_velocity_y + mean_velocity_z * mean_velocity_z

            for q in range(self.direction_count):
                # second-order Hermite projection of Pi: H_q = c_qa c_qb Pi_ab - trace / 3
                hermite_second_order = (self.lattice_velocities[q, 0] * self.lattice_velocities[q, 0] * stress_xx + self.lattice_velocities[q, 1] * self.lattice_velocities[q, 1] * stress_yy
                                        + self.lattice_velocities[q, 2] * self.lattice_velocities[q, 2] * stress_zz
                                        + 2.0 * (self.lattice_velocities[q, 0] * self.lattice_velocities[q, 1] * stress_xy
                                                 + self.lattice_velocities[q, 0] * self.lattice_velocities[q, 2] * stress_xz
                                                 + self.lattice_velocities[q, 1] * self.lattice_velocities[q, 2] * stress_yz)
                                        - (1.0 / 3.0) * trace)
                # first-order Hermite term (a1 = -F/2) that regularization drops; restores f_neq's
                # -F/2 first moment so the forced momentum balance matches the full-f scheme. F = (body_force_x, 0, 0)
                regularized_nonequilibrium = 4.5 * self.lattice_weights[q] * hermite_second_order - 1.5 * self.lattice_weights[q] * self.lattice_velocities[q, 0] * body_force_x

                # Guo source (BGK form)
                velocity_dot_direction = self.lattice_velocities[q, 0] * velocity_x + self.lattice_velocities[q, 1] * velocity_y + self.lattice_velocities[q, 2] * velocity_z
                force_dot_direction = self.lattice_velocities[q, 0] * body_force_x
                velocity_dot_force = velocity_x * body_force_x
                guo_source = guo_prefactor * self.lattice_weights[q] * (3.0 * (force_dot_direction - velocity_dot_force) + 9.0 * velocity_dot_direction * force_dot_direction)

                post_collision = self.feq(q, density, velocity_x, velocity_y, velocity_z, velocity_squared) + (1.0 - relaxation_rate) * regularized_nonequilibrium + guo_source
                if layer_strength_x > 0.0:
                    post_collision -= layer_strength_x * (post_collision - self.feq(q, 1.0, inlet_velocity, 0.0, 0.0, inlet_velocity * inlet_velocity))
                if layer_strength_z > 0.0:
                    post_collision -= layer_strength_z * (post_collision - self.feq(q, mean_density, mean_velocity_x, mean_velocity_y, mean_velocity_z, mean_velocity_squared))
                self.f[q, i, j, k] = post_collision

    def collide_reg(self, base_relaxation_time, smagorinsky_constant, body_force_x, inlet_velocity=0.0, mean_update_rate=0.0, z_layers_on=0):
        """
        Regularized collision; layers act only where sigma / sigma_z are nonzero (both default to 0).
        """

        self._collide_reg(base_relaxation_time, smagorinsky_constant, body_force_x, inlet_velocity, mean_update_rate, z_layers_on)

    @ti.kernel
    def collide_full(self, base_relaxation_time: ti.f32, smagorinsky_constant: ti.f32, body_force_x: ti.f32, trt: ti.i32):
        """
        One collision kernel: moments -> local tau (LES + wall model) -> TRT/BGK relax + Guo source.
        smagorinsky_constant = 0 disables LES, body_force_x = 0 disables forcing, trt = 0 gives BGK.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):

            # wall nodes only hold bounced populations for one step, don't collide them
            if self.solid[i, j, k] == 1 or self.lid[i, j, k] == 1:
                continue

            moments = self._moments(i, j, k, body_force_x)
            density = moments[0]
            velocity_x = moments[1]
            velocity_y = moments[2]
            velocity_z = moments[3]
            velocity_squared = moments[4]

            relaxation_time = base_relaxation_time
            if smagorinsky_constant > 0.0:
                stress = self._stress(i, j, k, density, velocity_x, velocity_y, velocity_z, velocity_squared)
                stress_xx = stress[0]
                stress_yy = stress[1]
                stress_zz = stress[2]
                stress_xy = stress[3]
                stress_xz = stress[4]
                stress_yz = stress[5]
                stress_magnitude = ti.sqrt(stress_xx * stress_xx + stress_yy * stress_yy + stress_zz * stress_zz + 2 * (stress_xy * stress_xy + stress_xz * stress_xz + stress_yz * stress_yz))
                relaxation_time = 0.5 * (base_relaxation_time + ti.sqrt(base_relaxation_time * base_relaxation_time + 18.0 * ti.sqrt(2.0) * smagorinsky_constant * smagorinsky_constant * stress_magnitude / density))

            relaxation_time += 3.0 * (self.nut_wall[i, j, k] + self.nut_les[i, j, k])

            # even (symmetric) and odd (antisymmetric) rates; trt = 0 -> BGK
            relaxation_rate_even = 1.0 / relaxation_time  # omega+
            relaxation_rate_odd = relaxation_rate_even  # omega-
            if trt == 1:
                relaxation_rate_odd = 1.0 / (0.5 + (3.0 / 16.0) / (relaxation_time - 0.5))  # magic parameter 3/16
            guo_prefactor_even = 1.0 - 0.5 * relaxation_rate_even
            guo_prefactor_odd = 1.0 - 0.5 * relaxation_rate_odd

            for q in range(self.direction_count):
                opposite = self.opposite_direction[q]
                # each pair once
                if q <= opposite:
                    velocity_dot_direction = self.lattice_velocities[q, 0] * velocity_x + self.lattice_velocities[q, 1] * velocity_y + self.lattice_velocities[q, 2] * velocity_z
                    equilibrium_even = self.lattice_weights[q] * density * (1 + 4.5 * velocity_dot_direction * velocity_dot_direction - 1.5 * velocity_squared)
                    equilibrium_odd = self.lattice_weights[q] * density * (3.0 * velocity_dot_direction)
                    force_dot_direction = self.lattice_velocities[q, 0] * body_force_x
                    velocity_dot_force = velocity_x * body_force_x
                    source_even = self.lattice_weights[q] * (9.0 * velocity_dot_direction * force_dot_direction - 3.0 * velocity_dot_force)
                    source_odd = self.lattice_weights[q] * (3.0 * force_dot_direction)
                    source_q = guo_prefactor_even * source_even + guo_prefactor_odd * source_odd
                    source_opposite = guo_prefactor_even * source_even - guo_prefactor_odd * source_odd
                    population_q = self.f[q, i, j, k]
                    population_opposite = self.f[opposite, i, j, k]
                    population_even = 0.5 * (population_q + population_opposite)
                    population_odd = 0.5 * (population_q - population_opposite)
                    self.f[q, i, j, k] = population_q - relaxation_rate_even * (population_even - equilibrium_even) - relaxation_rate_odd * (population_odd - equilibrium_odd) + source_q
                    if q != opposite:
                        self.f[opposite, i, j, k] = population_opposite - relaxation_rate_even * (population_even - equilibrium_even) + relaxation_rate_odd * (population_odd - equilibrium_odd) + source_opposite

    @ti.kernel
    def wall_model(self, viscosity: ti.f32, y1: ti.f32):
        """
        Log-law wall model on the full grid: at fluid nodes touching solid, set nut_wall so the first node
        carries tau_w = u_tau^2. Engages only at y+ > 30 (below that the node is resolved). Call after
        macroscopic(), before collide. y1 = first-node wall distance (0.5 for halfway bounce-back).
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.nut_wall[i, j, k] = 0.0
            if self.solid[i, j, k] == 0:
                # discrete inward normal: sum lattice vectors pointing at solid neighbours
                normal_sum_x = 0.0
                normal_sum_y = 0.0
                normal_sum_z = 0.0
                solid_neighbour_count = 0
                for q in range(self.direction_count):
                    neighbour_i = i + self.lattice_velocities[q, 0]
                    neighbour_j = j + self.lattice_velocities[q, 1]
                    neighbour_k = k + self.lattice_velocities[q, 2]
                    if 0 <= neighbour_i < self.nx and 0 <= neighbour_j < self.ny and 0 <= neighbour_k < self.nz:
                        if self.solid[neighbour_i, neighbour_j, neighbour_k] == 1:
                            normal_sum_x += self.lattice_velocities[q, 0]
                            normal_sum_y += self.lattice_velocities[q, 1]
                            normal_sum_z += self.lattice_velocities[q, 2]
                            solid_neighbour_count += 1
                normal_sum_magnitude = ti.sqrt(normal_sum_x * normal_sum_x + normal_sum_y * normal_sum_y + normal_sum_z * normal_sum_z)
                if solid_neighbour_count > 0 and normal_sum_magnitude > 0.0:
                    # normal into the fluid
                    normal_x = -normal_sum_x / normal_sum_magnitude
                    normal_y = -normal_sum_y / normal_sum_magnitude
                    normal_z = -normal_sum_z / normal_sum_magnitude
                    velocity_x = self.u[0, i, j, k]
                    velocity_y = self.u[1, i, j, k]
                    velocity_z = self.u[2, i, j, k]
                    # wall-parallel velocity
                    velocity_dot_normal = velocity_x * normal_x + velocity_y * normal_y + velocity_z * normal_z
                    parallel_x = velocity_x - velocity_dot_normal * normal_x
                    parallel_y = velocity_y - velocity_dot_normal * normal_y
                    parallel_z = velocity_z - velocity_dot_normal * normal_z
                    parallel_speed = ti.sqrt(parallel_x * parallel_x + parallel_y * parallel_y + parallel_z * parallel_z)
                    u_tau = self.wall_utau(parallel_speed, y1, viscosity)
                    y_plus = y1 * u_tau / viscosity
                    if y_plus > 30.0 and parallel_speed > 0.0:
                        self.nut_wall[i, j, k] = ti.max(0.0, u_tau * u_tau * y1 / parallel_speed - viscosity)

    @ti.kernel
    def _wall_model_fast(self, viscosity: ti.f32, scale: ti.f32):
        """
        Log-law wall model over the wall list (build_wall_list), with per-node wall distance, normal and
        wall velocity (scale * wall_uw, the inlet ramp). Same engagement rule as wall_model.
        """

        for m in range(self.n_wall):
            y1 = self.wall_y1[m]

            i = self.wall_ijk[m, 0]
            j = self.wall_ijk[m, 1]
            k = self.wall_ijk[m, 2]

            # velocity relative to the wall
            moments = self._moments(i, j, k, 0.0)
            velocity_x = moments[1] - scale * self.wall_uw[m, 0]
            velocity_y = moments[2] - scale * self.wall_uw[m, 1]
            velocity_z = moments[3] - scale * self.wall_uw[m, 2]

            # wall-parallel velocity
            normal_x = self.wall_n[m, 0]
            normal_y = self.wall_n[m, 1]
            normal_z = self.wall_n[m, 2]
            velocity_dot_normal = velocity_x * normal_x + velocity_y * normal_y + velocity_z * normal_z
            parallel_x = velocity_x - velocity_dot_normal * normal_x
            parallel_y = velocity_y - velocity_dot_normal * normal_y
            parallel_z = velocity_z - velocity_dot_normal * normal_z
            parallel_speed = ti.sqrt(parallel_x * parallel_x + parallel_y * parallel_y + parallel_z * parallel_z)
            u_tau = self.wall_utau(parallel_speed, y1, viscosity)
            y_plus = y1 * u_tau / viscosity
            wall_eddy_viscosity = 0.0
            if y_plus > 30.0 and parallel_speed > 0.0:
                wall_eddy_viscosity = ti.max(0.0, u_tau * u_tau * y1 / parallel_speed - viscosity)

            self.nut_wall[i, j, k] = wall_eddy_viscosity

    def wall_model_fast(self, viscosity, scale=1.0):
        """
        Wall-list log-law wall model; call build_wall_list first.
        """

        self._wall_model_fast(viscosity, scale)

    @ti.kernel
    def _les_wale(self, wale_constant: ti.f32, scale: ti.f32):
        """
        WALE subgrid viscosity from central / one-sided velocity gradients. Call after macroscopic().
        Solid neighbours contribute their wall velocity (scale * uw), so moving walls are not seen as a
        velocity jump. scale follows the inlet ramp like bounce_back.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.nut_les[i, j, k] = 0.0
            if self.solid[i, j, k] == 1 or self.lid[i, j, k] == 1:
                continue

            # neighbour velocities on each axis
            velocity_plus_x = self._uvec(i, j, k, 1, 0, 0, scale)
            velocity_minus_x = self._uvec(i, j, k, -1, 0, 0, scale)
            velocity_plus_y = self._uvec(i, j, k, 0, 1, 0, scale)
            velocity_minus_y = self._uvec(i, j, k, 0, -1, 0, scale)
            velocity_plus_z = self._uvec(i, j, k, 0, 0, 1, scale)
            velocity_minus_z = self._uvec(i, j, k, 0, 0, -1, scale)

            # difference spacing: 2 for central, 1 for one-sided at the domain edge
            spacing_x = ti.max(self._in_dom(i, j, k, 1, 0, 0) + self._in_dom(i, j, k, -1, 0, 0), 1)
            spacing_y = ti.max(self._in_dom(i, j, k, 0, 1, 0) + self._in_dom(i, j, k, 0, -1, 0), 1)
            spacing_z = ti.max(self._in_dom(i, j, k, 0, 0, 1) + self._in_dom(i, j, k, 0, 0, -1), 1)

            # velocity_gradient[a, b] = d u_a / d x_b
            velocity_gradient = ti.Matrix.zero(ti.f32, 3, 3)
            for a in ti.static(range(3)):
                velocity_gradient[a, 0] = (velocity_plus_x[a] - velocity_minus_x[a]) / spacing_x
                velocity_gradient[a, 1] = (velocity_plus_y[a] - velocity_minus_y[a]) / spacing_y
                velocity_gradient[a, 2] = (velocity_plus_z[a] - velocity_minus_z[a]) / spacing_z

            strain_rate = 0.5 * (velocity_gradient + velocity_gradient.transpose())  # S
            gradient_squared = velocity_gradient @ velocity_gradient  # g . g
            gradient_squared_trace = gradient_squared.trace()
            traceless_symmetric = 0.5 * (gradient_squared + gradient_squared.transpose()) - (gradient_squared_trace / 3.0) * ti.Matrix.identity(ti.f32, 3)  # S^d

            strain_contraction = (strain_rate * strain_rate).sum()  # S : S
            traceless_contraction = (traceless_symmetric * traceless_symmetric).sum()  # S^d : S^d

            epsilon = 1e-12
            traceless_root = ti.sqrt(traceless_contraction)
            strain_root = ti.sqrt(strain_contraction)
            numerator = traceless_contraction * traceless_root  # (S^d : S^d)^1.5
            denominator = strain_contraction * strain_contraction * strain_root + traceless_contraction * ti.sqrt(traceless_root) + epsilon  # (S : S)^2.5 + (S^d : S^d)^1.25
            self.nut_les[i, j, k] = wale_constant * wale_constant * numerator / denominator  # filter width = 1 cell

    def les_wale(self, wale_constant, scale=1.0):
        """
        WALE subgrid viscosity into nut_les.
        """

        self._les_wale(wale_constant, scale)

    @ti.func
    def _uvec(self, i, j, k, di, dj, dk, scale):
        """
        Velocity at neighbour (i + di, j + dj, k + dk): out of domain -> clamp to (i, j, k) (with les_wale's
        in-domain divisor this gives a one-sided difference); solid -> its wall velocity scale * uw
        (0 for static walls); else the stored velocity.

        Returns a 3-vector.
        """

        neighbour_i = i + di
        neighbour_j = j + dj
        neighbour_k = k + dk
        velocity = ti.Vector([0.0, 0.0, 0.0])
        if neighbour_i < 0 or neighbour_i >= self.nx or neighbour_j < 0 or neighbour_j >= self.ny or neighbour_k < 0 or neighbour_k >= self.nz:
            velocity = ti.Vector([self.u[0, i, j, k], self.u[1, i, j, k], self.u[2, i, j, k]])
        elif self.solid[neighbour_i, neighbour_j, neighbour_k] == 1:
            velocity = scale * ti.Vector([self.uw[0, neighbour_i, neighbour_j, neighbour_k], self.uw[1, neighbour_i, neighbour_j, neighbour_k], self.uw[2, neighbour_i, neighbour_j, neighbour_k]])
        else:
            velocity = ti.Vector([self.u[0, neighbour_i, neighbour_j, neighbour_k], self.u[1, neighbour_i, neighbour_j, neighbour_k], self.u[2, neighbour_i, neighbour_j, neighbour_k]])

        return velocity

    @ti.func
    def _in_dom(self, i, j, k, di, dj, dk) -> ti.i32:
        """
        Whether (i + di, j + dj, k + dk) lies inside the domain.

        Returns 1 inside, else 0.
        """

        neighbour_i = i + di
        neighbour_j = j + dj
        neighbour_k = k + dk
        inside = 1
        if neighbour_i < 0 or neighbour_i >= self.nx or neighbour_j < 0 or neighbour_j >= self.ny or neighbour_k < 0 or neighbour_k >= self.nz:
            inside = 0

        return inside

    def build_wall_list(self, phi=None):
        """
        List the fluid nodes touching solid, with inward normal, wall distance y1 and mean wall velocity.
        Without phi: lattice-sum normals and y1 = 0.5; with an SDF phi: exact distance and SDF normals.
        """

        solid = self.solid.to_numpy()
        lattice_velocities = d3q19.lattice_velocities
        nx, ny, nz = self.nx, self.ny, self.nz

        def shift(array, offset):
            # array[c] -> array[c + offset], zero-filled out of bounds
            shifted = np.zeros_like(array)
            source_slices = tuple(slice(max(step, 0), array.shape[axis] + min(step, 0)) for axis, step in enumerate(offset))
            destination_slices = tuple(slice(max(-step, 0), array.shape[axis] - max(step, 0)) for axis, step in enumerate(offset))
            shifted[destination_slices] = array[source_slices]

            return shifted

        # sum of lattice vectors pointing at solid neighbours, and the neighbour count
        normal_sum = np.zeros((3, nx, ny, nz))
        solid_neighbour_count = np.zeros((nx, ny, nz), np.int32)
        for q in range(self.direction_count):
            neighbour_solid = shift(solid, lattice_velocities[q])  # 1 where the c_q neighbour is solid, in bounds
            solid_neighbour_count += neighbour_solid
            for axis in range(3):
                normal_sum[axis] += lattice_velocities[q, axis] * neighbour_solid

        # wall velocity of each solid neighbour
        wall_velocity = self.uw.to_numpy()
        wall_velocity_sum = np.zeros((3, nx, ny, nz), np.float32)
        for q in range(self.direction_count):
            for axis in range(3):
                wall_velocity_sum[axis] += shift(wall_velocity[axis] * solid, lattice_velocities[q])

        normal_sum_magnitude = np.sqrt((normal_sum * normal_sum).sum(axis=0))
        wall_nodes_mask = (solid == 0) & (solid_neighbour_count > 0) & (normal_sum_magnitude > 0.0)
        wall_nodes = np.argwhere(wall_nodes_mask).astype(np.int32)  # (M, 3)
        normals = (-normal_sum[:, wall_nodes_mask] / normal_sum_magnitude[wall_nodes_mask]).T.astype(np.float32)  # (M, 3)
        mean_wall_velocity = (wall_velocity_sum[:, wall_nodes_mask] / solid_neighbour_count[wall_nodes_mask]).T.astype(np.float32)  # (M, 3) mean wall velocity seen by each node

        wall_count = wall_nodes.shape[0]
        self.n_wall = wall_count

        wall_distance = np.full(wall_count, 0.5, np.float32)  # halfway bounce-back default
        if phi is not None:
            points_x = wall_nodes[:, 0].astype(np.float64)
            points_y = wall_nodes[:, 1].astype(np.float64)
            points_z = wall_nodes[:, 2].astype(np.float64)
            wall_distance = phi(points_x, points_y, points_z).astype(np.float32)  # true normal distance to the surface
            normals = normals_from_sdf(phi, points_x, points_y, points_z).astype(np.float32)

        self.wall_ijk = ti.field(ti.i32, shape=(wall_count, 3))
        self.wall_ijk.from_numpy(wall_nodes)
        self.wall_n = ti.field(ti.f32, shape=(wall_count, 3))
        self.wall_n.from_numpy(normals)
        self.wall_y1 = ti.field(ti.f32, shape=wall_count)
        self.wall_y1.from_numpy(wall_distance)
        self.wall_uw = ti.field(ti.f32, shape=(wall_count, 3))
        self.wall_uw.from_numpy(mean_wall_velocity)

    @ti.kernel
    def _stream(self):
        """
        Pull each population from its upstream neighbour into f_new, periodic wrap, parallel over destination cells.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            for q in range(self.direction_count):
                # where this population came from; % nx / ny / nz = periodic wrap
                source_i = (i - self.lattice_velocities[q, 0]) % self.nx
                source_j = (j - self.lattice_velocities[q, 1]) % self.ny
                source_k = (k - self.lattice_velocities[q, 2]) % self.nz
                self.f_new[q, i, j, k] = self.f[q, source_i, source_j, source_k]

    def stream(self):
        """
        Stream kernel, then copy the buffer back into f (plain Python, not a kernel).
        """

        self._stream()
        self.f.copy_from(self.f_new)

    @ti.kernel
    def _bounce_back(self, scale: ti.f32):
        """
        No-slip wall: reverse the populations at solid nodes; moving walls (uw != 0) add the Ladd momentum
        correction 6 w_q (c_q . u_w). scale multiplies every wall speed (inlet ramp).
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.solid[i, j, k] == 1:
                wall_velocity_x = scale * self.uw[0, i, j, k]
                wall_velocity_y = scale * self.uw[1, i, j, k]
                wall_velocity_z = scale * self.uw[2, i, j, k]

                for q in range(self.direction_count):
                    opposite = self.opposite_direction[q]

                    if q < opposite:
                        swap = self.f[q, i, j, k]
                        self.f[q, i, j, k] = self.f[opposite, i, j, k]
                        self.f[opposite, i, j, k] = swap

                if wall_velocity_x != 0.0 or wall_velocity_y != 0.0 or wall_velocity_z != 0.0:
                    for q in range(self.direction_count):
                        neighbour_i = (i + self.lattice_velocities[q, 0]) % self.nx
                        neighbour_j = (j + self.lattice_velocities[q, 1]) % self.ny
                        neighbour_k = (k + self.lattice_velocities[q, 2]) % self.nz

                        # link leads into the fluid
                        if self.solid[neighbour_i, neighbour_j, neighbour_k] == 0:
                            direction_dot_wall_velocity = self.lattice_velocities[q, 0] * wall_velocity_x + self.lattice_velocities[q, 1] * wall_velocity_y + self.lattice_velocities[q, 2] * wall_velocity_z
                            self.f[q, i, j, k] += 6.0 * self.lattice_weights[q] * direction_dot_wall_velocity

    def bounce_back(self, scale=1.0):
        """
        Full-way bounce-back on solid nodes, with the moving-wall term scaled by scale.
        """

        self._bounce_back(scale)

    def set_wall_fractions(self, q):
        """
        Load the per-link wall fractions q (Q, nx, ny, nz) and build the compact boundary-link list.

        Bouzidi and drag_interp loop over this list instead of scanning all 19 x N entries each step.
        Call after body is loaded; link_part is taken from it.
        """

        self.q.from_numpy(q)
        direction, i, j, k = np.nonzero(q > 0.0)
        count = len(direction)
        size = max(count, 1)
        self.link_count = count
        self.link_direction = ti.field(ti.i32, shape=size)
        self.link_node = ti.field(ti.i32, shape=(size, 3))
        self.link_fraction = ti.field(ti.f32, shape=size)
        self.link_part = ti.field(ti.i32, shape=size)
        if count > 0:
            self.link_direction.from_numpy(direction.astype(np.int32))
            self.link_node.from_numpy(np.stack([i, j, k], axis=1).astype(np.int32))
            self.link_fraction.from_numpy(q[direction, i, j, k].astype(np.float32))

            # part id at the solid end of each link (body must be loaded before this call)
            body = self.body.to_numpy()
            solid_i = (i + d3q19.lattice_velocities[direction, 0]) % self.nx
            solid_j = (j + d3q19.lattice_velocities[direction, 1]) % self.ny
            solid_k = (k + d3q19.lattice_velocities[direction, 2]) % self.nz
            self.link_part.from_numpy(body[solid_i, solid_j, solid_k].astype(np.int32))

    @ti.func
    def _link_wall_velocity(self, i, j, k, direction, fraction, scale):
        """
        Wall velocity at the wall point x_f + q c_d, linear between the link ends
        (exact for rigid-body motion when uw is set on both sides of the surface).

        Returns a 3-vector.
        """

        solid_i = (i + self.lattice_velocities[direction, 0]) % self.nx
        solid_j = (j + self.lattice_velocities[direction, 1]) % self.ny
        solid_k = (k + self.lattice_velocities[direction, 2]) % self.nz
        wall_velocity = ti.Vector([0.0, 0.0, 0.0])
        for axis in ti.static(range(3)):
            wall_velocity[axis] = scale * ((1.0 - fraction) * self.uw[axis, i, j, k] + fraction * self.uw[axis, solid_i, solid_j, solid_k])

        return wall_velocity

    @ti.kernel
    def _bounce_back_interp(self, scale: ti.f32, link_direction: ti.template(), link_node: ti.template(),
                            link_fraction: ti.template(), count: ti.i32):
        """
        Bouzidi wall with moving-wall term: delta = 6 w (c_opposite . u_w), divided by 2q on the q > 1/2
        branch (Bouzidi et al. 2001). Loops over the boundary-link list. Reads fc, writes f.
        """

        for m in range(count):
            direction = link_direction[m]
            i = link_node[m, 0]
            j = link_node[m, 1]
            k = link_node[m, 2]
            fraction = link_fraction[m]
            opposite = self.opposite_direction[direction]
            population_incoming = self.fc[direction, i, j, k]  # post-collision f_i at x_f
            wall_velocity = self._link_wall_velocity(i, j, k, direction, fraction, scale)
            wall_momentum = 6.0 * self.lattice_weights[opposite] * (self.lattice_velocities[opposite, 0] * wall_velocity[0] + self.lattice_velocities[opposite, 1] * wall_velocity[1] + self.lattice_velocities[opposite, 2] * wall_velocity[2])

            if fraction <= 0.5:
                upstream_i = i - self.lattice_velocities[direction, 0]
                upstream_j = j - self.lattice_velocities[direction, 1]
                upstream_k = k - self.lattice_velocities[direction, 2]
                upstream_fluid = 0
                if 0 <= upstream_i < self.nx and 0 <= upstream_j < self.ny and 0 <= upstream_k < self.nz:
                    if self.solid[upstream_i, upstream_j, upstream_k] == 0:
                        upstream_fluid = 1

                if upstream_fluid == 1:
                    population_upstream = self.fc[direction, upstream_i, upstream_j, upstream_k]
                    self.f[opposite, i, j, k] = 2.0 * fraction * population_incoming + (1.0 - 2.0 * fraction) * population_upstream + wall_momentum
                # no fluid upstream node (domain edge or thin gap) -> fall back to halfway
                else:
                    self.f[opposite, i, j, k] = population_incoming + wall_momentum
            else:
                population_opposite = self.fc[opposite, i, j, k]  # post-collision f_ibar at x_f
                inverse_two_q = 1.0 / (2.0 * fraction)
                self.f[opposite, i, j, k] = population_incoming * inverse_two_q + (2.0 * fraction - 1.0) * inverse_two_q * population_opposite + wall_momentum * inverse_two_q

    def bounce_back_interp(self, scale=1.0):
        """
        Bouzidi interpolated bounce-back over the boundary-link list; call set_wall_fractions first.
        """

        self._bounce_back_interp(scale, self.link_direction, self.link_node, self.link_fraction, self.link_count)

    @ti.kernel
    def free_slip_y(self):
        """
        Free-slip y walls: specular reflection, mirrors the y component.
        """

        for i, k in ti.ndrange(self.nx, self.nz):
            for q in range(self.direction_count):
                mirrored = self.mirror_y[q]
                if q < mirrored:
                    # bottom wall j = 0
                    swap_bottom = self.f[q, i, 0, k]
                    self.f[q, i, 0, k] = self.f[mirrored, i, 0, k]
                    self.f[mirrored, i, 0, k] = swap_bottom
                    # top wall j = ny - 1
                    swap_top = self.f[q, i, self.ny - 1, k]
                    self.f[q, i, self.ny - 1, k] = self.f[mirrored, i, self.ny - 1, k]
                    self.f[mirrored, i, self.ny - 1, k] = swap_top

    @ti.kernel
    def free_slip_z(self):
        """
        Free-slip z walls: specular reflection, mirrors the z component.
        """

        for i, j in ti.ndrange(self.nx, self.ny):
            for q in range(self.direction_count):
                mirrored = self.mirror_z[q]
                if q < mirrored:
                    # wall k = 0
                    swap_bottom = self.f[q, i, j, 0]
                    self.f[q, i, j, 0] = self.f[mirrored, i, j, 0]
                    self.f[mirrored, i, j, 0] = swap_bottom
                    # wall k = nz - 1
                    swap_top = self.f[q, i, j, self.nz - 1]
                    self.f[q, i, j, self.nz - 1] = self.f[mirrored, i, j, self.nz - 1]
                    self.f[mirrored, i, j, self.nz - 1] = swap_top

    @ti.kernel
    def inlet_neem(self, inlet_velocity: ti.f32):
        """
        Guo non-equilibrium extrapolation inlet: equilibrium at the imposed velocity with density taken from
        x = 1, plus the neighbour non-equilibrium part. Holds the free stream rigidly, but diverges where two
        free-slip walls meet.
        """

        for j, k in ti.ndrange(self.ny, self.nz):
            self._neem_column(j, k, inlet_velocity)

    @ti.kernel
    def inlet_neem_open(self, inlet_velocity: ti.f32):
        """
        Same as inlet_neem but drives open columns only, leaving solid at the inlet plane alone.
        """

        for j, k in ti.ndrange(self.ny, self.nz):
            if self.solid[0, j, k] == 0 and self.solid[1, j, k] == 0:
                self._neem_column(j, k, inlet_velocity)

    @ti.func
    def _neem_column(self, j, k, inlet_velocity):
        """
        Set the inlet populations at (0, j, k) from feq(rho at x = 1, inlet_velocity) + the regularized x = 1 non-equilibrium.
        """

        moments = self._moments(1, j, k, 0.0)
        neighbour_density = moments[0]
        velocity_x = moments[1]
        velocity_y = moments[2]
        velocity_z = moments[3]
        velocity_squared = moments[4]
        stress = self._stress(1, j, k, neighbour_density, velocity_x, velocity_y, velocity_z, velocity_squared)  # [Pxx, Pyy, Pzz, Pxy, Pxz, Pyz]
        trace = stress[0] + stress[1] + stress[2]

        for q in range(self.direction_count):
            direction_x = self.lattice_velocities[q, 0]
            direction_y = self.lattice_velocities[q, 1]
            direction_z = self.lattice_velocities[q, 2]
            hermite_second_order = (direction_x * direction_x * stress[0] + direction_y * direction_y * stress[1] + direction_z * direction_z * stress[2]
                                    + 2.0 * (direction_x * direction_y * stress[3] + direction_x * direction_z * stress[4] + direction_y * direction_z * stress[5])
                                    - (1.0 / 3.0) * trace)
            boundary_equilibrium = self.feq(q, neighbour_density, inlet_velocity, 0.0, 0.0, inlet_velocity * inlet_velocity)
            self.f[q, 0, j, k] = boundary_equilibrium + 4.5 * self.lattice_weights[q] * hermite_second_order

    @ti.kernel
    def outlet(self):
        """
        Zero-gradient outlet: copy the second-to-last plane onto the last.
        """

        for j, k in ti.ndrange(self.ny, self.nz):
            for q in range(self.direction_count):
                self.f[q, self.nx - 1, j, k] = self.f[q, self.nx - 2, j, k]

    @ti.kernel
    def outlet_pressure(self, outlet_density: ti.f32):
        """
        Pressure (density) outlet, Guo non-equilibrium extrapolation: impose outlet_density at x = nx - 1, take
        the velocity and the non-equilibrium part from x = nx - 2. Pins the mean density, which a velocity
        inlet + zero-gradient outlet leave free to drift. Open columns only.
        """

        for j, k in ti.ndrange(self.ny, self.nz):
            if self.solid[self.nx - 1, j, k] == 0 and self.solid[self.nx - 2, j, k] == 0:
                i = self.nx - 2
                moments = self._moments(i, j, k, 0.0)
                neighbour_density = moments[0]
                velocity_x = moments[1]
                velocity_y = moments[2]
                velocity_z = moments[3]
                velocity_squared = moments[4]

                stress = self._stress(i, j, k, neighbour_density, velocity_x, velocity_y, velocity_z, velocity_squared)
                trace = stress[0] + stress[1] + stress[2]

                for q in range(self.direction_count):
                    direction_x = self.lattice_velocities[q, 0]
                    direction_y = self.lattice_velocities[q, 1]
                    direction_z = self.lattice_velocities[q, 2]
                    hermite_second_order = (direction_x * direction_x * stress[0] + direction_y * direction_y * stress[1] + direction_z * direction_z * stress[2]
                                            + 2.0 * (direction_x * direction_y * stress[3] + direction_x * direction_z * stress[4] + direction_y * direction_z * stress[5])
                                            - (1.0 / 3.0) * trace)
                    boundary_equilibrium = self.feq(q, outlet_density, velocity_x, velocity_y, velocity_z, velocity_squared)
                    self.f[q, self.nx - 1, j, k] = boundary_equilibrium + 4.5 * self.lattice_weights[q] * hermite_second_order

    @ti.kernel
    def sponge_relax(self, inlet_velocity: ti.f32):
        """
        Separate relaxation absorbing layer for collide_full / collide_trt cases (collide_reg fuses it):
        f -> f - sigma (f - feq(1, inlet_velocity, 0, 0)) near x = 0 and x = nx - 1.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            layer_strength = self.sigma[i, k]
            if layer_strength > 0.0 and self.solid[i, j, k] == 0:
                for q in range(self.direction_count):
                    target = self.feq(q, 1.0, inlet_velocity, 0.0, 0.0, inlet_velocity * inlet_velocity)
                    self.f[q, i, j, k] -= layer_strength * (self.f[q, i, j, k] - target)

    @ti.kernel
    def sponge_relax_mean(self, mean_update_rate: ti.f32):
        """
        Separate z-wall absorbing layer (superseded by the fused layers in collide_reg; kept for test 32):
        relax f toward the local running mean (rho_bar, u_bar), not the free stream.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            layer_strength = self.sigma_z[i, k]
            if layer_strength > 0.0 and self.solid[i, j, k] == 0:
                moments = self._moments(i, j, k, 0.0)
                self.rho_bar[i, j, k] += mean_update_rate * (moments[0] - self.rho_bar[i, j, k])
                for axis in ti.static(range(3)):
                    self.u_bar[axis, i, j, k] += mean_update_rate * (moments[1 + axis] - self.u_bar[axis, i, j, k])
                mean_density = self.rho_bar[i, j, k]
                mean_velocity_x = self.u_bar[0, i, j, k]
                mean_velocity_y = self.u_bar[1, i, j, k]
                mean_velocity_z = self.u_bar[2, i, j, k]
                mean_velocity_squared = mean_velocity_x * mean_velocity_x + mean_velocity_y * mean_velocity_y + mean_velocity_z * mean_velocity_z
                for q in range(self.direction_count):
                    target = self.feq(q, mean_density, mean_velocity_x, mean_velocity_y, mean_velocity_z, mean_velocity_squared)
                    self.f[q, i, j, k] -= layer_strength * (self.f[q, i, j, k] - target)

    @ti.kernel
    def drag(self):
        """
        Force on all solid by momentum exchange, sum 2 c_q f_q over fluid -> solid links. Call after collide, before stream.
        """

        self._drag_mask(0)

    @ti.kernel
    def _drag_interp(self, scale: ti.f32, link_direction: ti.template(), link_node: ti.template(),
                     link_fraction: ti.template(), link_part: ti.template(), count: ti.i32):
        """
        Momentum exchange for interpolated walls over the boundary-link list, Galilean-invariant form
        (Wen et al. 2014): F = sum c_d (f_in + f_out) - u_w (f_in - f_out), u_w at the wall point.
        Call after bounce_back_interp, so f holds the reconstructed reflection.
        """

        for axis in range(self.dimension):
            self.force[axis] = 0.0
        for part, axis in ti.ndrange(self.max_parts + 1, self.dimension):
            self.part_force[part, axis] = 0.0
        for m in range(count):
            direction = link_direction[m]
            i = link_node[m, 0]
            j = link_node[m, 1]
            k = link_node[m, 2]
            part = link_part[m]
            opposite = self.opposite_direction[direction]
            population_in = self.fc[direction, i, j, k]  # post-collision, into the wall
            population_out = self.f[opposite, i, j, k]  # reconstructed reflection
            wall_velocity = self._link_wall_velocity(i, j, k, direction, link_fraction[m], scale)
            for axis in ti.static(range(3)):
                contribution = (population_in + population_out) * self.lattice_velocities[direction, axis] - wall_velocity[axis] * (population_in - population_out)
                self.force[axis] += contribution
                if part > 0:
                    self.part_force[part, axis] += contribution

    def drag_interp(self, scale=1.0):
        """
        Galilean-invariant momentum-exchange force for Bouzidi walls, into force.
        """

        self._drag_interp(scale, self.link_direction, self.link_node, self.link_fraction, self.link_part, self.link_count)

    @ti.func
    def _drag_mask(self, use_body: ti.i32):
        """
        Momentum-exchange force; use_body = 1 sums only body links (Cd), 0 sums all solid links.
        Staircase walls, static bodies only (moving parts use Bouzidi + drag_interp).
        """

        for axis in range(self.dimension):
            self.force[axis] = 0.0
        for part, axis in ti.ndrange(self.max_parts + 1, self.dimension):
            self.part_force[part, axis] = 0.0
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if self.solid[i, j, k] == 0:
                for q in range(self.direction_count):
                    neighbour_i = i + self.lattice_velocities[q, 0]
                    neighbour_j = j + self.lattice_velocities[q, 1]
                    neighbour_k = k + self.lattice_velocities[q, 2]
                    if 0 <= neighbour_i < self.nx and 0 <= neighbour_j < self.ny and 0 <= neighbour_k < self.nz:
                        part = 0
                        link_hits_target = False
                        if use_body:
                            part = self.body[neighbour_i, neighbour_j, neighbour_k]
                            link_hits_target = part > 0
                        else:
                            link_hits_target = self.solid[neighbour_i, neighbour_j, neighbour_k] == 1
                        if link_hits_target:
                            momentum_exchange = 2.0 * self.f[q, i, j, k]
                            for axis in ti.static(range(3)):
                                contribution = momentum_exchange * self.lattice_velocities[q, axis]
                                self.force[axis] += contribution
                                if part > 0:
                                    self.part_force[part, axis] += contribution

    @ti.kernel
    def drag_body(self):
        """
        Same as drag but only links into body parts (body > 0), so tunnel walls don't pollute Cd; also fills part_force per part id.
        Call after collide, before stream.
        """

        self._drag_mask(1)

    @ti.kernel
    def f_absmax(self) -> ti.f32:
        """
        Blow-up probe: max |f| over the field, forced huge on any NaN. A cheap GPU reduction so the health
        check never pulls the whole f array on the healthy path.

        Returns max |f| (1e30 on NaN).
        """

        max_magnitude = 0.0
        for q, i, j, k in self.f:
            value = self.f[q, i, j, k]
            # NaN is the only value not equal to itself
            if value != value:
                ti.atomic_max(max_magnitude, 1e30)
            else:
                ti.atomic_max(max_magnitude, ti.abs(value))

        return max_magnitude

    @ti.func
    def feq(self, q, density, velocity_x, velocity_y, velocity_z, velocity_squared):
        """
        D3Q19 equilibrium, the one copy collide_full / collide_reg / _init_eq all use.

        Returns feq_q.
        """

        velocity_dot_direction = self.lattice_velocities[q, 0] * velocity_x + self.lattice_velocities[q, 1] * velocity_y + self.lattice_velocities[q, 2] * velocity_z

        return self.lattice_weights[q] * density * (1 + 3 * velocity_dot_direction + 4.5 * velocity_dot_direction * velocity_dot_direction - 1.5 * velocity_squared)

    @ti.func
    def _moments(self, i, j, k, body_force_x):
        """
        Density and velocity at (i, j, k), with the Guo half-force shift in x.

        Returns Vector[density, velocity_x, velocity_y, velocity_z, velocity_squared].
        """

        density = 0.0
        momentum_x = 0.0
        momentum_y = 0.0
        momentum_z = 0.0
        for q in range(self.direction_count):
            population = self.f[q, i, j, k]
            density += population
            momentum_x += population * self.lattice_velocities[q, 0]
            momentum_y += population * self.lattice_velocities[q, 1]
            momentum_z += population * self.lattice_velocities[q, 2]
        velocity_x = (momentum_x + 0.5 * body_force_x) / density
        velocity_y = momentum_y / density
        velocity_z = momentum_z / density
        velocity_squared = velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z

        return ti.Vector([density, velocity_x, velocity_y, velocity_z, velocity_squared])

    @ti.func
    def _stress(self, i, j, k, density, velocity_x, velocity_y, velocity_z, velocity_squared):
        """
        Non-equilibrium stress Pi_ab = sum c_qa c_qb (f_q - feq_q) at (i, j, k).

        Returns Vector[Pxx, Pyy, Pzz, Pxy, Pxz, Pyz].
        """

        stress_xx = 0.0
        stress_yy = 0.0
        stress_zz = 0.0
        stress_xy = 0.0
        stress_xz = 0.0
        stress_yz = 0.0
        for q in range(self.direction_count):
            nonequilibrium = self.f[q, i, j, k] - self.feq(q, density, velocity_x, velocity_y, velocity_z, velocity_squared)
            stress_xx += nonequilibrium * self.lattice_velocities[q, 0] * self.lattice_velocities[q, 0]
            stress_yy += nonequilibrium * self.lattice_velocities[q, 1] * self.lattice_velocities[q, 1]
            stress_zz += nonequilibrium * self.lattice_velocities[q, 2] * self.lattice_velocities[q, 2]
            stress_xy += nonequilibrium * self.lattice_velocities[q, 0] * self.lattice_velocities[q, 1]
            stress_xz += nonequilibrium * self.lattice_velocities[q, 0] * self.lattice_velocities[q, 2]
            stress_yz += nonequilibrium * self.lattice_velocities[q, 1] * self.lattice_velocities[q, 2]

        return ti.Vector([stress_xx, stress_yy, stress_zz, stress_xy, stress_xz, stress_yz])

    @ti.func
    def wall_utau(self, parallel_speed, y1, viscosity):
        """
        Device twin of turbulence/wall_function.friction_velocity: Newton inversion of the log law.

        Returns u_tau (0 for non-positive speed).
        """

        u_tau = 0.0
        if parallel_speed > 0.0:
            von_karman = 0.41
            log_law_constant = 5.2
            u_tau = ti.sqrt(viscosity * parallel_speed / y1)  # viscous initial guess
            inverse_von_karman = 1.0 / von_karman
            # bound must be a compile-time constant
            for _ in range(50):
                log_term = inverse_von_karman * ti.log(y1 * u_tau / viscosity) + log_law_constant
                residual = u_tau * log_term - parallel_speed
                if ti.abs(residual) < 1e-8:
                    break
                residual_derivative = log_term + inverse_von_karman
                u_tau_next = u_tau - residual / residual_derivative
                u_tau = u_tau_next if u_tau_next > 0.0 else u_tau * 0.5

        return u_tau

    @ti.kernel
    def _init_eq(self):
        """
        Fill f with feq from whatever is in rho and u.
        """

        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            density = self.rho[i, j, k]
            velocity_x = self.u[0, i, j, k]
            velocity_y = self.u[1, i, j, k]
            velocity_z = self.u[2, i, j, k]
            velocity_squared = velocity_x * velocity_x + velocity_y * velocity_y + velocity_z * velocity_z

            for q in range(self.direction_count):
                self.f[q, i, j, k] = self.feq(q, density, velocity_x, velocity_y, velocity_z, velocity_squared)

    def init_equilibrium(self, velocity_x, velocity_y, velocity_z, density=1.0):
        """
        Start from equilibrium at a prescribed velocity field (numpy arrays in, fields loaded, kernel fills f).
        """

        self.u.from_numpy(np.stack([velocity_x, velocity_y, velocity_z]).astype(np.float32))
        self.rho.from_numpy(np.full((self.nx, self.ny, self.nz), density, np.float32))
        self._init_eq()
