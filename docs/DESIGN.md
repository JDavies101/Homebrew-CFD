# Design & Roadmap

This document is the technical plan. It is meant to be edited as decisions change.

## 1. Constraints that drive every decision

- **CFD is memory-bandwidth bound, not compute bound.** The 3090's ~936 GB/s bandwidth
  and 24 GB VRAM are the real limits. A D3Q19 LBM lattice in FP32 costs roughly
  `19 × 4 bytes × 2 (double buffer) ≈ 152 bytes/cell`. 24 GB therefore caps a naive
  implementation near ~150M cells; FP16 storage with FP32 math roughly doubles that.
  Domain-size budgeting is a first-class concern, not an afterthought.
- **Explicit + local wins on the GPU.** LBM's streaming/collision steps touch only
  nearest neighbors and need no global pressure solve, so they saturate bandwidth well.
- **Accuracy comes from resolution near walls + turbulence modeling**, both of which are
  expensive. This is the fidelity/cost tension the project manages deliberately.

## 2. Physics & numerics

- **Method:** Lattice Boltzmann, D3Q19 to start (good accuracy/cost balance); D3Q27
  option later for higher-fidelity turbulence.
- **Collision operator:** start with BGK (single relaxation time) for clarity and
  testing; move to MRT or a regularized/entropic operator for stability at the high
  Reynolds numbers of car aero (Re ~ 10^6).
- **Turbulence:** LES with a Smagorinsky (or WALE) subgrid model. Direct wall resolution
  is infeasible at these Re on one GPU, so use **wall functions**.
- **Boundary conditions:** inlet velocity, outlet pressure/outflow, no-slip walls
  (bounce-back), and — essential for cars — a **moving ground plane** and **rotating
  wheel** boundaries. Getting these right is what separates real automotive aero from a
  generic box-in-a-tunnel.
- **Precision:** FP32 baseline; evaluate FP16/mixed storage to extend domain size once
  the FP32 path is validated.

## 3. Geometry pipeline

Everything funnels into one common step — **voxelize into the lattice** (solid/fluid
flagging + sub-voxel wall distance for the wall model). That is LBM's big ergonomics win:
no body-fitted meshing, so any source that can produce a watertight surface or a filled
2D outline works. Three input paths feed it:

1. **Standard/parametric shapes** — airfoils via NACA 4/5-digit generators and `.dat`
   coordinate import (UIUC database), extended to multi-element F1 wing sections. Fully
   in-code, no external files; these also serve as validation geometry. *Available first.*
2. **Hand-drawn shapes** — sketch or import a 2D cross-section (drawing canvas, or PNG →
   threshold → contour). Natural fit for the 2D solver; extrude for 3D. *Phase 1–2.*
3. **3D CAD (SolidWorks)** — export **STL** from SolidWorks (native, watertight), not
   STEP: STL is triangle soup that voxelizes directly with no CAD kernel dependency.
   OBJ later. *Phase 4.*

After voxelization, assign per-part tags (front wing, floor, wheels…) so forces can be
broken out by part.

> No CAD/geometry files exist yet — intentional. Phases 1–3 run entirely on parametric and
> hand-drawn shapes, so validation never blocks on external assets. STL import is only
> needed at Phase 4.

## 4. Outputs (all four requested)

- **Force coefficients:** integrate momentum exchange on solid nodes → Cd, Cl/downforce,
  drag counts; per-part breakdown via geometry tags.
- **Field visualization:** pressure, velocity, vorticity, streamlines; **VTK export** for
  ParaView.
- **Live/interactive view:** in-framework real-time render (Taichi has a built-in GUI);
  adjust inlet speed / yaw angle on the fly at reduced resolution.
- **Design comparison:** config-driven runs + a sweep harness (ride height, wing angle,
  yaw) that tabulates coefficients across variants.

## 5. Validation strategy (the heart of "accuracy first")

We do not trust the solver on an F1 shape until it passes a ladder of cases with known
answers. Two test tiers:

**Tier A — unit tests (mechanics, fast, run every commit):**
- Equilibrium distribution sums to correct density/momentum at a single node.
- Mass conservation over a closed periodic domain (no leaks).
- Symmetry: symmetric setup produces symmetric fields.
- Bounce-back wall gives zero velocity at the wall.
- Relaxation drives a perturbed node toward equilibrium at the expected rate.

**Tier B — physics validation (published reference data):**
| Case | What it checks | Reference |
|------|----------------|-----------|
| Poiseuille / Couette flow | Viscosity, wall BCs | Analytic parabolic/linear profile |
| Lid-driven cavity | 2D recirculation accuracy | Ghia, Ghia & Shin (1982) |
| Flow past a cylinder | Drag + vortex shedding | Cd vs Re; Strouhal ≈ 0.2 |
| Backward-facing step | Separation/reattachment length | Armaly et al. |
| **Ahmed body** | **Automotive bluff-body wake, Cd** | **Standard car-aero benchmark** |

The Ahmed body is the gate: it is the canonical automotive validation case. Only after it
matches published Cd and wake structure do we run real F1 geometry. OpenFOAM provides an
independent FVM cross-check on selected cases.

## 6. Repository layout (planned)

```
homebrew-cfd/
├── README.md
├── docs/               # design, validation notes, results
├── src/
│   ├── lbm/            # core solver: lattice, collision, streaming, BCs
│   ├── geometry/       # STL import, voxelization, part tagging
│   ├── turbulence/     # LES subgrid model, wall functions
│   ├── post/           # force integration, VTK export, viewer
│   └── config/         # run configuration
├── cases/              # validation + demo case definitions
├── tests/              # pytest: Tier A unit + Tier B validation
└── requirements.txt
```

## 7. Phased roadmap

**Phase 0 — Scaffold.** Repo structure, environment (Taichi + CUDA verified on the 3090),
CI-style `pytest` skeleton, config loader. *Exit: `pytest` runs green on an empty suite.*

**Phase 1 — 2D LBM core. ✅ DONE.** D2Q9 BGK in **NumPy** (not Taichi — Phase 1 stays
CPU/NumPy for clarity while learning the physics; GPU port lands in Phase 2 where 3D needs
it). Operators: moments, equilibrium, BGK collision, streaming, bounce-back walls, moving
wall, Guo body force. Validated:
- **Poiseuille**: profile exactly parabolic (R²=1.0), peak within 1% of analytic (walls
  taken from the fit to avoid sub-cell ambiguity). Guo forcing gives 2nd-order accuracy.
- **Lid-driven cavity vs Ghia et al. (Re=100)**: correct vortex + centerline structure;
  interior minimum matches Ghia within ~5% at 64² and **<1% at 128²** (convergent).

37 tests (Tier A unit + Tier B validation); validation cases marked `slow`.
*Exit met.*

**Phase 2 — 3D + first bluff body. 🔶 IN PROGRESS.** D3Q19 engine (`Simulation3D`), Guo
forcing, Zou-He/NEEM velocity inlet, zero-gradient outlet, free-slip walls, cylinder as a
voxelized obstacle, drag via momentum exchange. VTK export pending. Cylinder result:
- **Flow validated.** Rigid NEEM inlet holds the free-stream (measured Re=101 vs nominal
  100); vortex shedding **Strouhal = 0.165** at Re=100 (ref ~0.16–0.18); **no shedding at
  Re=40** (correct sub-critical behavior).
- **Drag: correct method, known discretization offset.** Momentum-exchange force is
  `Σ 2 cᵢ fᵢ` (post-collision, over fluid→solid links). Cd ≈ 1.9 vs unbounded ref ~1.4 —
  the ~35% is the documented over-prediction of **full-way (staircase) bounce-back** at
  finite resolution, which converges with resolution and is cured by **interpolated
  bounce-back** (the accuracy upgrade, deferred to Phase 3 with the wall-treatment work;
  needed for smooth aero surfaces regardless). Note the two references differ: unbounded
  cylinder Cd≈1.4 vs the confined Schäfer–Turek channel benchmark Cd≈3.2 — free-slip walls
  target the unbounded value.

Sphere (true 3D): runs stably at low Re (Re≈15–20; BGK needs high τ, so higher Re awaits
MRT in Phase 3). Cd lands in the staircase over-prediction band, but the free-slip **corner**
where the y and z symmetry planes meet injects spurious energy (`max|u|` ~4× inlet) — a
node-based specular-reflection artifact needing proper corner handling (Phase 3). So the
sphere validates the machinery, not a clean Cd.

Regression net: `tests/test_engine3d.py` (6 Tier-A invariant tests — macroscopic, collide
fixed-point, conservation, bounce-back swap, free-slip involution, inlet). It caught a real
top-wall bug in `free_slip_y`/`_z` (a reused temp), which also slightly cleaned the cylinder
Cd (1.89→1.83). 41 fast tests total; validation cases marked `slow`.

*Exit (remaining): VTK export. Deferred to Phase 3: interpolated bounce-back (staircase Cd),
free-slip corner treatment (sphere), MRT/regularized collision (high-Re stability).*

**Phase 3 — Turbulence + walls.** LES subgrid model, wall functions; backward-facing step
and **Ahmed body** validation. *Exit: Ahmed body Cd and wake match published data.*

**Phase 4 — Automotive features.** STL import + voxelization of real parts, moving ground,
rotating wheels, per-part force breakdown. *Exit: front-wing or full-car run with sane,
stable coefficients.*

**Phase 5 — Interactivity & sweeps.** Live viewer, parameter sweeps, A/B comparison
tables, mixed-precision for larger domains.

Each phase ends only when its validation gate passes. Docs and tests are updated within
the same phase, not after.

## 8. Additional lattice stencils (future)

The lattice is a pure-data descriptor (`Q, D, E, W, OPP, CS2`) that the operators
consume generically, so new stencils are data-only additions — no kernel changes for
same-dimension sets. Candidates, and what each is actually good for:

- **D3Q15 / D3Q19 / D3Q27** — the 3D Navier–Stokes workhorses. D3Q19 is the default;
  D3Q27 for high-Re isotropy/stability; D3Q15 for cheap/coarse runs (weaker isotropy).
  *(D3Q19 done; Q15/Q27 planned as data-only additions.)*
- **D2Q9** — the standard 2D NS stencil (done).
- **D2Q5 / D3Q7** — *not* full fluid stencils. These are advection–diffusion lattices for
  a scalar field (temperature, species). Useful later if we add heat transfer or passive
  scalars alongside the flow — a second distribution on a small stencil.
- **Higher-order 2D (D2Q17, D2Q37) / 3D (D3Q39, D3Q41)** — needed only for thermal
  (compressible/high-Mach) or high-accuracy work. Overkill for incompressible aero;
  revisit only if a case demands it.

Note on "any DnQm": more velocities is not automatically better — a stencil must satisfy
the isotropy moment conditions (`Σ wᵢeᵢ=0`, `Σ wᵢeᵢ⊗eᵢ=cs²I`, and the 4th-order condition)
to reproduce the target physics. Arbitrary DnQm sets (e.g. a made-up D2Q16/D2Q25) generally
do **not** — only specific, derived velocity/weight sets work. Add stencils from the
literature, not by picking a velocity count. Every new stencil ships with the same
descriptor tests (weights sum to 1, opposites reverse, isotropy moments).

## 9. Open questions to revisit

- Taichi vs Warp final call (decide after Phase 1 ergonomics).
- MRT vs regularized/entropic collision for high-Re stability (decide in Phase 3).
- FP16 storage — how much accuracy do we trade for domain size? (measure in Phase 5).
- Real F1 geometry source and its licensing (needed by Phase 4).
- Cylinder Cd calibration: blockage, resolution, τ, MEM factor (Phase 2, in progress).
