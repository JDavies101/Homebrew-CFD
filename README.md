# Homebrew CFD - GPU Lattice-Boltzmann CFD Engine

A from-scratch computational fluid dynamics engine for incompressible and low-Mach flow,
built on the Lattice Boltzmann Method (LBM). It runs on a single workstation
(RTX 3090, Ryzen 5800X3D) and is validated against established benchmarks before it is applied
to real geometry. External aerodynamics, and Formula 1 in particular, is the flagship
application, but the solver is general: external bodies, internal ducts, heat transfer, and
aeroacoustics are all within reach of the same core. Anthropic's Claude was used to assist with
code outline, documentation upkeep, and planning. Early icon concepts were drafted with Google Gemini; the final icon, splash screen and ribbon icons are generated in code (the streamlines are a computed potential flow around the wing), and the wordmark uses Saira under the SIL Open Font License.

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
- **Application:** PySide6 (Qt) with PyVista (VTK) for the 3D viewport and pyqtgraph for live
  monitors; packaged with PyInstaller and Inno Setup.
- **Cross-validation (later):** OpenFOAM.
- **Tests:** pytest for solver mechanics and physics validation cases.

## Status

Phases 0 through 3 are complete, Phase 4 is largely complete, the operating envelope is mapped, and a
first working desktop application (Phase 6 MVP) builds into a Windows installer. The result is a
validated 2D and 3D GPU solver with turbulence, sub-cell walls, moving ground and rotating wheels,
STL import, per-part forces, and an application that sets up, runs and monitors cases.

- **Phase 1 - 2D core (NumPy):** D2Q9 BGK covering moments, equilibrium, collision, streaming,
  bounce-back, moving wall, and Guo body force. Validated against Poiseuille flow (exact
  parabola, R^2 = 1, peak within 1%) and the Ghia et al. lid-driven cavity (within 1% at
  128^2).
- **Phase 2 - 3D on the GPU (Taichi):** the D3Q19 engine (`Simulation3D`) with Guo forcing,
  velocity inlet and outlet, free-slip walls, voxelized obstacles, momentum-exchange drag, VTK
  export, and a live progress and divergence monitor. Validated against cylinder flow and sphere
  drag. The same code runs on CPU or CUDA via a single flag.
- **Phase 3 - turbulence and walls:** TRT collision, Bouzidi interpolated bounce-back on
  curved walls, LES (Smagorinsky, later WALE), regularized collision, and a log-law wall function,
  composed into one stable high-Reynolds-number stack. Validated across a backward-facing step,
  the Re_tau = 180 turbulent channel (law of the wall, U+ = 18.5), and the Ahmed body, stable to
  Re_H = 30000. The relative aerodynamic gate passed: a square-versus-round nose comparison
  yields a correct 48% drag increase with the expected separation mechanism.
- **Phase 4 - automotive and general geometry:** WALE subgrid model, relaxation layers, signed
  distance field geometry, STL import validated against the analytic sphere (Cd within 0.1%),
  moving ground, rotating wheels (Taylor-Couette and Magnus checks), per-part force breakdown,
  and an inverted NACA 4412 wing in ground effect whose downforce-versus-ride-height trend
  reproduces the experimental enhancement-then-loss shape. A full 3D front wing or car remains.
- **Operating envelope:** relaxation-time floor (0.501 regularized), resolution, blockage, Mach
  number (U <= 0.05 incompressible; Strouhal 0.171 with no Mach trend), Ahmed resolution, wing
  Reynolds number (the ground-effect peak moves toward the ground as Re rises), and relative
  accuracy under refinement (curve shape preserved, absolute level within 4-5%).
- **Generic runner:** every 3D example runs through one `Case` description and one time loop
  (`run_case`), and each port reproduces its logged reference run exactly.
- **Phase 6 MVP - desktop application:** versioned JSON case files, a command-line solve into
  self-contained run folders (progress, results, live force samples, clean stop), and a PySide6
  application with a setup tree, property forms with live validation, a 3D viewport with named
  views, run control with a console and live force monitors, and autosave. Packaged with
  PyInstaller and Inno Setup into a Windows installer that reproduces the reference wing run.

Known limits are documented in [`docs/DESIGN.md`](docs/DESIGN.md): absolute Ahmed Cd is not
grid-converged on 24 GB, and the quasi-2D laminar wing is a qualitative, not experimental-regime,
comparison.

## Roadmap

Full detail is in [`docs/DESIGN.md`](docs/DESIGN.md). In summary:

- **v0.1.1 (next release):** UPX off and a VirusTotal re-check, installer built by GitHub Actions
  with checksums and build-provenance attestation, pinned and audited dependencies, a code-signing
  decision, and the UI backlog below. Plan in [`docs/DESIGN.md`](docs/DESIGN.md) (Phase 6).
- **Application polish and release pipeline:** UI backlog (ribbon tabs, results tab, warmup
  marker and running mean on monitors, frozen inputs while running), GitHub Actions builds from
  tagged commits with checksums and build-provenance attestation, and a validation suite that
  runs before every release.
- **Application features:** STL import in the UI with pre-run voxel and link preview, memory and
  time estimates, run comparison, parametric sweeps (ride height, incidence) as one queued study,
  checkpoints, and more templates (Ahmed body, sphere, wheel).
- **Phase 4 completion:** a 3D front wing or full car run.
- **Platforms:** backend auto-detect (CUDA, Vulkan, CPU; Metal later), Linux build, then D3Q15 /
  D3Q27 stencils and a 2D engine with the same physics as the 3D one.
- **Phase 5 - interactivity and sweeps:** live in-run visualization and mixed precision for
  larger domains.
- **Future branch - multiphase and free-surface**, as specific cases require.

### Running it

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate      # Taichi requires Python 3.12 or earlier
pip install -r requirements.txt
python -m src.config.environment          # verify CUDA on the GPU
python -m app                             # desktop application (File > New from template)
python -m src.run cases/templates/wing_ground.json   # headless solve into runs/<date>_<name>/
python -m src.examples.cylinder           # validation example: prints Cd and Strouhal
pytest -m "not slow"                      # fast unit tests
pytest                                    # plus physics validation (Poiseuille, cavity, and others)
powershell -ExecutionPolicy Bypass -File packaging/build.ps1   # Windows app folder + installer
```

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture, numerics, and full roadmap.

## License

[MIT](LICENSE) - free to use, learn from, and build on.
