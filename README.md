# Homebrew CFD - GPU Lattice-Boltzmann CFD Engine

A from-scratch computational fluid dynamics engine for incompressible and low-Mach flow,
built on the Lattice Boltzmann Method (LBM). It runs on a single workstation
(RTX 3090, Ryzen 5800X3D) and is validated against established benchmarks before it is applied
to real geometry. External aerodynamics, and Formula 1 in particular, is the flagship
application, but the solver is general: external bodies, internal ducts, heat transfer, and
aeroacoustics are all within reach of the same core.

## Goals

- **Accuracy first.** Every capability is backed by a validation case with published
  reference data. The solver is trusted only where the benchmarks support it.
- **Built by hand.** The core solver is written directly in Python on a GPU compute
  framework rather than wrapped around an existing package.
- **Fits the hardware.** 24 GB of VRAM is a hard ceiling, and the design treats memory
  bandwidth as the primary constraint throughout.

## Method

The solver core is the Lattice Boltzmann Method. LBM is explicit, local, and highly
parallel, which maps to the GPU more naturally than a pressure-solve Navier-Stokes code, and
it is a production-grade approach: PowerFLOW, the industry-standard automotive and F1 aero
solver, is LBM-based. Geometry is voxelized directly into the lattice, avoiding the
body-fitted meshing that finite-volume methods require, so a new shape is a mesh import rather
than a new solver setup. Turbulence is handled with a Large-Eddy Simulation subgrid model and
wall functions. OpenFOAM is reserved for later use as an independent finite-volume
cross-check, not as a second hand-written solver.

## Applications

The same core addresses a range of incompressible and low-Mach problems, ordered by the
additional machinery each requires on top of the current engine:

- **External aerodynamics of arbitrary bodies** (following STL import): cars and F1
  components, trucks, cyclists, drones and UAVs, buildings and bridges (wind loading), and
  sports-ball aerodynamics. The same solver applies to each; only the geometry changes.
- **Internal low-speed flow:** ducts, manifolds, HVAC, intake and exhaust systems.
- **Heat transfer** (via a second, thermal distribution): forced and natural convection,
  conjugate heat transfer, and electronics cooling.
- **Aeroacoustics:** LBM is transient and low-dissipation, so it resolves near-body pressure
  fluctuations directly; far-field radiation is handled through an acoustic analogy
  (Ffowcs-Williams-Hawkings).
- **Multiphase and free-surface flow** (a future branch, via Shan-Chen or color-gradient
  models): sloshing, sprays, and hydrodynamics. A substantial extension, tracked but not
  near-term.

**Out of scope by design.** Standard LBM is low-Mach and incompressible (Ma below
approximately 0.3). High-speed compressible flow, shocks, and reacting or detonation flow
belong to a different class of solver and are deliberately not pursued here; the
validation-first charter favors stating this limit over stretching past it.

## Accuracy expectation

Certification-grade CFD uses meshes of 100M to over 1B cells on compute clusters. On a single
3090 the target is trustworthy relative comparison (whether one wing outperforms another, the
effect of ride height, which geometry sheds less) together with credible absolute trends,
rather than certification-grade numbers. Broadening the range of problems does not move this
ceiling, which is set by hardware; the validation harness makes the boundary explicit for
every case.

## Stack

- **Language:** Python 3.12 (Taichi requires 3.12 or earlier).
- **GPU kernels:** [Taichi](https://www.taichi-lang.org/) with the CUDA backend on the 3090.
  NVIDIA Warp is the fallback if Taichi proves limiting.
- **Post-processing:** NumPy, with VTK export for [ParaView](https://www.paraview.org/).
- **Interactivity (planned):** a Taichi GGUI live viewer for in-run visualization, followed
  by a standalone setup, run-control, and post-processing application as a dedicated phase.
- **Cross-validation (later):** OpenFOAM.
- **Tests:** pytest for solver mechanics and physics validation cases.

## Status

Phases 0 through 3 are complete; Phase 4 (automotive and general geometry) is next. The result
is a validated 2D and 3D GPU solver with turbulence, sub-cell walls, and a wall-modeled
high-Reynolds-number stack, developed and verified one operator at a time.

- **Phase 1 - 2D core (NumPy):** D2Q9 BGK covering moments, equilibrium, collision, streaming,
  bounce-back, moving wall, and Guo body force. Validated against Poiseuille flow (exact
  parabola, R^2 = 1, peak within 1%) and the Ghia et al. lid-driven cavity (within 1% at
  128^2).
- **Phase 2 - 3D on the GPU (Taichi):** the D3Q19 engine (`Simulation3D`) with Guo forcing,
  velocity inlet and outlet, free-slip walls, voxelized obstacles, momentum-exchange drag, VTK
  export, and a live progress and divergence monitor. Validated against cylinder flow (correct
  Reynolds number and vortex-shedding Strouhal near 0.165, with no shedding below the critical
  Reynolds number) and sphere drag. The same code runs on CPU or CUDA via a single flag.
- **Phase 3 - turbulence and walls:** TRT collision, Bouzidi interpolated bounce-back on
  curved walls (cylinder Cd 1.576, sphere on sub-cell walls), LES Smagorinsky, regularized
  collision, and a generalized log-law wall function, composed into one stable
  high-Reynolds-number stack. Validated across a backward-facing step (reattachment
  x_r/S = 2.47), the Re_tau = 180 turbulent channel (law of the wall, U+ = 18.5), and the
  Ahmed body, stable to Re_H = 30000. The relative aerodynamic gate passed: a square-versus-
  round nose comparison yields a correct 48% drag increase with the expected separation
  mechanism.

Known limits are documented in [`docs/DESIGN.md`](docs/DESIGN.md): absolute Ahmed Cd is beyond
reach on 24 GB and would require a cluster, and several research-grade components (a
wall-aware subgrid model, free-slip corner treatment, and an SDF wall-distance field) are
carried into Phase 4, where real geometry requires them.

## Roadmap

Full detail is in [`docs/DESIGN.md`](docs/DESIGN.md). In summary:

- **Phase 4 - Automotive and general geometry:** SDF wall-distance and wall-aware subgrid
  foundations, STL import and voxelization (validated by reducing an STL sphere to the analytic
  result), moving ground and rotating wheels, per-part force breakdown, and a front-wing or
  full-car run.
- **Phase 5 - Interactivity and sweeps:** a Taichi GGUI live viewer, parameter sweeps, A/B
  comparison tables, and mixed precision for larger domains.
- **Phase 6 - Application (UI):** a standalone setup, run-control, and post-processing
  application.
- **Future branch - multiphase and free-surface**, as specific cases require.

### Running it

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate      # Taichi requires Python 3.12 or earlier
pip install -r requirements.txt
python -m src.config.environment          # verify CUDA on the GPU
python -m src.examples.cylinder           # 3D cylinder: prints Cd and Strouhal
pytest -m "not slow"                      # fast unit tests
pytest                                    # plus physics validation (Poiseuille, cavity, and others)
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture, numerics, and full roadmap.

## License

[MIT](LICENSE) - free to use, learn from, and build on.
