# Homebrew CFD - GPU Wind Tunnel Simulator

A from-scratch computational fluid dynamics wind tunnel for external aerodynamics -
F1 cars, subassemblies (front wings, floors, diffusers), or arbitrary shapes - built to
run on a single workstation (RTX 3090, Ryzen 5800X3D) and validated against known
benchmarks before it is trusted on real geometry.

## Goals

- **Accuracy first.** Every capability is backed by a validation case with published
  reference data. We trust the solver only where the benchmarks say we should.
- **Learn by building.** The core solver is written by hand in Python on a GPU compute
  framework - not wrapped from a black box.
- **Runs on my hardware.** 24 GB of VRAM is the hard ceiling; the design treats memory
  bandwidth as the primary constraint.

## Approach in one paragraph

The solver core is the **Lattice Boltzmann Method (LBM)**. LBM is explicit, local, and
embarrassingly parallel, so it maps to the GPU far better than a traditional
pressure-solve Navier-Stokes code - and it is not a toy: **PowerFLOW, the industry-standard
automotive/F1 aero solver, is LBM.** Geometry is voxelized directly into the lattice, which
sidesteps the painful body-fitted meshing that FVM requires. Turbulence is handled with a
Large-Eddy Simulation (LES) subgrid model plus wall functions, and car cases add a moving
ground and rotating wheels. A traditional finite-volume solver (OpenFOAM) is used **later,
as an independent cross-check**, not as a second hand-written solver.

## Honest accuracy expectation

Real F1 CFD uses 100M-1B+ cell meshes on clusters. On one 3090 we target **trustworthy
relative comparisons** (does wing A beat wing B; what does ride height do) and credible
absolute drag/downforce trends - not certification-grade numbers. The validation harness
makes this boundary explicit.

## Stack

- **Language:** Python 3.
- **GPU kernels:** [Taichi](https://www.taichi-lang.org/) (CUDA backend on the 3090).
  NVIDIA Warp is the fallback if Taichi proves limiting.
- **Post-processing:** NumPy, VTK export for [ParaView](https://www.paraview.org/);
  lightweight live viewer for interactive runs.
- **Cross-validation (later):** OpenFOAM.
- **Tests:** pytest for solver mechanics + physics validation cases.

## Status

**Phases 0-2 complete, Phase 3 underway: a validated 2D and 3D GPU solver with
turbulence and sub-cell walls.** Built and checked one operator at a time.

- **Phase 1 - 2D core (NumPy):** D2Q9 BGK - moments, equilibrium, collision, streaming,
  bounce-back, moving wall, Guo body force. Validated against **Poiseuille** (exact
  parabola, R^2=1, peak <1%) and the **Ghia et al. lid-driven cavity** (<1% at 128^2).
- **Phase 2 - 3D on the GPU (Taichi):** D3Q19 engine (`Simulation3D`), Guo forcing,
  velocity inlet/outlet, free-slip walls, voxelized obstacles, drag via momentum exchange,
  VTK export, and a live progress/ETA + divergence monitor. Validated against **cylinder**
  flow (measured Re and vortex-shedding Strouhal ~ 0.165 both correct; no shedding below
  the critical Re) and **sphere** drag. Same code runs on CPU or CUDA via one flag.

- **Phase 3 - turbulence and walls (in progress):** **TRT** collision (stability decoupled
  from viscosity - the sphere now runs at Re=50 where BGK diverged above Re=20),
  **interpolated (Bouzidi) bounce-back** for curved walls (cylinder Cd **1.83 -> 1.576**
  toward the ~1.4 reference), and **LES Smagorinsky** subgrid turbulence. All collision
  variants are one composable kernel. Each was validated by reducing exactly to the
  previous scheme.

Honest limits, all documented in [`docs/DESIGN.md`](docs/DESIGN.md): the sphere still uses
staircase bounce-back (no `wall_fraction_sphere` yet) so its Cd is over-predicted, and
absolute drag remains sensitive to blockage and resolution. Still to come: **wall
functions**, then the **backward-facing step** and **Ahmed body** gate before F1 geometry.

### Run it

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate      # Taichi needs Python <=3.12
pip install -r requirements.txt
python -m src.config.environment          # verify CUDA on the GPU
python -m src.examples.cylinder           # 3D cylinder: prints Cd + Strouhal
pytest -m "not slow"                      # fast unit tests
pytest                                    # + physics validation (Poiseuille, cavity)
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture, numerics, and full roadmap.

## License

[MIT](LICENSE) - free to use, learn from, and build on.
