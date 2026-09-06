# Homebrew CFD — GPU Wind Tunnel Simulator

A from-scratch computational fluid dynamics wind tunnel for external aerodynamics —
F1 cars, subassemblies (front wings, floors, diffusers), or arbitrary shapes — built to
run on a single workstation (RTX 3090, Ryzen 5800X3D) and validated against known
benchmarks before it is trusted on real geometry.

## Goals

- **Accuracy first.** Every capability is backed by a validation case with published
  reference data. We trust the solver only where the benchmarks say we should.
- **Learn by building.** The core solver is written by hand in Python on a GPU compute
  framework — not wrapped from a black box.
- **Runs on my hardware.** 24 GB of VRAM is the hard ceiling; the design treats memory
  bandwidth as the primary constraint.

## Approach in one paragraph

The solver core is the **Lattice Boltzmann Method (LBM)**. LBM is explicit, local, and
embarrassingly parallel, so it maps to the GPU far better than a traditional
pressure-solve Navier–Stokes code — and it is not a toy: **PowerFLOW, the industry-standard
automotive/F1 aero solver, is LBM.** Geometry is voxelized directly into the lattice, which
sidesteps the painful body-fitted meshing that FVM requires. Turbulence is handled with a
Large-Eddy Simulation (LES) subgrid model plus wall functions, and car cases add a moving
ground and rotating wheels. A traditional finite-volume solver (OpenFOAM) is used **later,
as an independent cross-check**, not as a second hand-written solver.

## Honest accuracy expectation

Real F1 CFD uses 100M–1B+ cell meshes on clusters. On one 3090 we target **trustworthy
relative comparisons** (does wing A beat wing B; what does ride height do) and credible
absolute drag/downforce trends — not certification-grade numbers. The validation harness
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

Planning. No solver code yet. See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture
and the phased roadmap.
