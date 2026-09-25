# Homebrew CFD - GPU Lattice-Boltzmann CFD Engine

A from-scratch computational fluid dynamics engine for incompressible / low-Mach flow,
built on the **Lattice Boltzmann Method (LBM)**, running on a single workstation
(RTX 3090, Ryzen 5800X3D) and validated against known benchmarks before it is trusted on
real geometry. External aerodynamics - and F1 in particular - is the flagship demonstrator,
but the solver is general: any body in a flow, internal ducts, heat transfer, and
aeroacoustics are all in reach of the same core.

## Goals

- **Accuracy first.** Every capability is backed by a validation case with published
  reference data. We trust the solver only where the benchmarks say we should.
- **Learn by building.** The core solver is written by hand in Python on a GPU compute
  framework - not wrapped from a black box.
- **Runs on my hardware.** 24 GB of VRAM is the hard ceiling; the design treats memory
  bandwidth as the primary constraint.

## Approach in one paragraph

The solver core is the **Lattice Boltzmann Method**. LBM is explicit, local, and
embarrassingly parallel, so it maps to the GPU far better than a traditional
pressure-solve Navier-Stokes code - and it is not a toy: **PowerFLOW, the industry-standard
automotive/F1 aero solver, is LBM.** Geometry is voxelized directly into the lattice, which
sidesteps the painful body-fitted meshing that FVM requires - so a new shape is a new mesh
import, not a new solver. Turbulence is handled with a Large-Eddy Simulation (LES) subgrid
model plus wall functions. A traditional finite-volume solver (OpenFOAM) is used **later, as
an independent cross-check**, not as a second hand-written solver.

## What it is for

The same core handles a wide range of incompressible / low-Mach problems. Ordered by how
much new machinery each needs on top of today's engine:

- **External aerodynamics of any body** (once STL import lands): cars and F1 parts, trucks,
  cyclists, drones/UAVs, buildings and bridges (wind loading), sports-ball aero. Same solver,
  different geometry.
- **Internal low-speed flow:** ducts, manifolds, HVAC, intake/exhaust.
- **Heat transfer** (a second, thermal distribution): forced/natural convection, conjugate
  heat transfer, electronics cooling.
- **Aeroacoustics:** LBM is transient and low-dissipation, so it resolves near-body pressure
  fluctuations directly; far-field uses an acoustic analogy (Ffowcs-Williams-Hawkings).
- **Multiphase / free-surface** (a future branch, Shan-Chen or color-gradient): sloshing,
  sprays, hydro. A major extension, tracked but not near-term.

**Out of scope, by design.** Standard LBM is low-Mach and incompressible (Ma < ~0.3). High-
speed compressible flow, shocks, and reacting/detonation flow are a fundamentally different
solver class - this engine is not the tool for them, and the validation-first charter means
we say so rather than stretch it.

## Honest accuracy expectation

Certification-grade CFD uses 100M-1B+ cell meshes on clusters. On one 3090 we target
**trustworthy relative comparisons** (does wing A beat wing B; what does ride height do; which
geometry sheds less) and credible absolute trends - not certification numbers. Broadening the
range of *problems* does not move this ceiling: it is set by hardware, and the validation
harness makes the boundary explicit for every case.

## Stack

- **Language:** Python 3.12 (Taichi needs <= 3.12).
- **GPU kernels:** [Taichi](https://www.taichi-lang.org/) (CUDA backend on the 3090).
  NVIDIA Warp is the fallback if Taichi proves limiting.
- **Post-processing:** NumPy, VTK export for [ParaView](https://www.paraview.org/).
- **Interactivity (planned):** a Taichi GGUI live viewer for in-run visualization, then a
  standalone setup / run-control / post app (geometry load, boundary assignment, launch,
  A/B compare) as its own phase.
- **Cross-validation (later):** OpenFOAM.
- **Tests:** pytest for solver mechanics + physics validation cases.

## Status

**Phases 0-3 complete; Phase 4 (automotive / general geometry) next.** A validated 2D and 3D
GPU solver with turbulence, sub-cell walls, and a wall-modeled high-Re stack, built and
checked one operator at a time.

- **Phase 1 - 2D core (NumPy):** D2Q9 BGK - moments, equilibrium, collision, streaming,
  bounce-back, moving wall, Guo body force. Validated against **Poiseuille** (exact parabola,
  R^2=1, peak <1%) and the **Ghia et al. lid-driven cavity** (<1% at 128^2).
- **Phase 2 - 3D on the GPU (Taichi):** D3Q19 engine (`Simulation3D`), Guo forcing,
  velocity inlet / outlet, free-slip walls, voxelized obstacles, drag via momentum exchange,
  VTK export, live progress/ETA + divergence monitor. Validated against **cylinder** (Re and
  vortex-shedding Strouhal ~0.165 correct; no shedding below critical Re) and **sphere** drag.
  Same code runs on CPU or CUDA via one flag.
- **Phase 3 - turbulence and walls:** **TRT** collision, **Bouzidi** interpolated bounce-back
  on curved walls (cylinder Cd **1.576**, sphere on sub-cell walls), **LES Smagorinsky**,
  **regularized collision**, and a **generalized log-law wall function**, all composed into
  one stable high-Re stack. Validated across a **backward-facing step** (reattachment
  x_r/S = 2.47), the **Re_tau=180 turbulent channel** (law of the wall, U+ = 18.5), and the
  **Ahmed body** stable to Re_H = 30000. The relative aero gate passed: a **square vs round
  nose** comparison gives a correct **+48% drag** with the right separation mechanism.

Honest limits, all in [`docs/DESIGN.md`](docs/DESIGN.md): absolute Ahmed Cd is out of reach on
24 GB (needs a cluster); a few research-grade pieces (wall-aware SGS, free-slip corner
treatment, SDF wall distance) are carried into Phase 4, where real geometry needs them.

## Roadmap

Full detail in [`docs/DESIGN.md`](docs/DESIGN.md). In short:

- **Phase 4 - Automotive / general geometry:** SDF wall-distance + wall-aware SGS foundations,
  **STL import + voxelization** (validated by reducing an STL sphere to the analytic result),
  moving ground + rotating wheels, per-part force breakdown, then a front-wing or full-car run.
- **Phase 5 - Interactivity & sweeps:** Taichi GGUI live viewer, parameter sweeps, A/B tables,
  mixed precision for larger domains.
- **Phase 6 - Application (UI):** standalone setup / run-control / post-processing app.
- **Future branch - multiphase / free-surface**, when a case calls for it.

### Run it

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate      # Taichi needs Python <= 3.12
pip install -r requirements.txt
python -m src.config.environment          # verify CUDA on the GPU
python -m src.examples.cylinder           # 3D cylinder: prints Cd + Strouhal
pytest -m "not slow"                      # fast unit tests
pytest                                    # + physics validation (Poiseuille, cavity, ...)
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture, numerics, and full roadmap.

## License

[MIT](LICENSE) - free to use, learn from, and build on.
