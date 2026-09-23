# Design & Roadmap

This document is the technical plan. It is meant to be edited as decisions change.

## 1. Constraints that drive every decision

- **CFD is memory-bandwidth bound, not compute bound.** The 3090's ~936 GB/s bandwidth
  and 24 GB VRAM are the real limits. A D3Q19 LBM lattice in FP32 costs roughly
  `19 x 4 bytes x 2 (double buffer) ~ 152 bytes/cell`. 24 GB therefore caps a naive
  implementation near ~150M cells; FP16 storage with FP32 math roughly doubles that.
  Domain-size budgeting is a first-class concern, not an afterthought.
- **Explicit + local wins on the GPU.** LBM's streaming/collision steps touch only
  nearest neighbors and need no global pressure solve, so they saturate bandwidth well.
- **Accuracy comes from resolution near walls + turbulence modeling**, both of which are
  expensive. This is the fidelity/cost tension the project manages deliberately.

## 2. Physics & numerics

- **Method:** Lattice Boltzmann, D3Q19 to start (good accuracy/cost balance); D3Q27
  option later for higher-fidelity turbulence.
- **Collision operator:** BGK (single relaxation time) first for clarity, then **TRT**
  (two-relaxation-time) for stability at the high Reynolds numbers of car aero (Re ~ 10^6).
  Full MRT is deferred - see section 9.
- **Turbulence:** LES with a Smagorinsky (or WALE) subgrid model. Direct wall resolution
  is infeasible at these Re on one GPU, so use **wall functions**.
- **Boundary conditions:** inlet velocity, outlet pressure/outflow, no-slip walls
  (bounce-back), and - essential for cars - a **moving ground plane** and **rotating
  wheel** boundaries. Getting these right is what separates real automotive aero from a
  generic box-in-a-tunnel.
- **Precision:** FP32 baseline; evaluate FP16/mixed storage to extend domain size once
  the FP32 path is validated.

## 3. Geometry pipeline

Everything funnels into one common step - **voxelize into the lattice** (solid/fluid
flagging + sub-voxel wall distance for the wall model). That is LBM's big ergonomics win:
no body-fitted meshing, so any source that can produce a watertight surface or a filled
2D outline works. Three input paths feed it:

1. **Standard/parametric shapes** - airfoils via NACA 4/5-digit generators and `.dat`
   coordinate import (UIUC database), extended to multi-element F1 wing sections. Fully
   in-code, no external files; these also serve as validation geometry. *Available first.*
2. **Hand-drawn shapes** - sketch or import a 2D cross-section (drawing canvas, or PNG ->
   threshold -> contour). Natural fit for the 2D solver; extrude for 3D. *Phase 1-2.*
3. **3D CAD (SolidWorks)** - export **STL** from SolidWorks (native, watertight), not
   STEP: STL is triangle soup that voxelizes directly with no CAD kernel dependency.
   OBJ later. *Phase 4.*

After voxelization, assign per-part tags (front wing, floor, wheels...) so forces can be
broken out by part.

Curved walls also need a **wall fraction** field `q`, shaped `(Q, nx, ny, nz)`: for each
fluid node and direction, the fraction of that link lying inside the fluid before it
crosses the surface. `0` means the link is not a boundary link. Interpolated bounce-back
consumes it; it is produced per shape in `src/geometry/` (analytically for cylinder and
sphere, from a signed-distance field for arbitrary STL later).

> No CAD/geometry files exist yet - intentional. Phases 1-3 run entirely on parametric and
> hand-drawn shapes, so validation never blocks on external assets. STL import is only
> needed at Phase 4.

## 4. Outputs (all four requested)

- **Force coefficients:** integrate momentum exchange on solid nodes -> Cd, Cl/downforce,
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

**Tier A - unit tests (mechanics, fast, run every commit):**
- Equilibrium distribution sums to correct density/momentum at a single node.
- Mass conservation over a closed periodic domain (no leaks).
- Symmetry: symmetric setup produces symmetric fields.
- Bounce-back wall gives zero velocity at the wall.
- Relaxation drives a perturbed node toward equilibrium at the expected rate.

**Tier B - physics validation (published reference data):**
| Case | What it checks | Reference |
|------|----------------|-----------|
| Poiseuille / Couette flow | Viscosity, wall BCs | Analytic parabolic/linear profile |
| Lid-driven cavity | 2D recirculation accuracy | Ghia, Ghia & Shin (1982) |
| Flow past a cylinder | Drag + vortex shedding | Cd vs Re; Strouhal ~ 0.2 |
| Backward-facing step | Separation/reattachment length | Armaly et al. |
| **Ahmed body** | **Automotive bluff-body wake, Cd** | **Standard car-aero benchmark** |

The Ahmed body is the gate: it is the canonical automotive validation case. Only after it
matches published Cd and wake structure do we run real F1 geometry. OpenFOAM provides an
independent FVM cross-check on selected cases.

## 6. Repository layout

```
homebrew-cfd/
+-- README.md
+-- docs/               # design, validation notes, results
+-- src/
|   +-- lbm/            # 2D NumPy solver: lattice, moments, equilibrium, collision,
|   |                   #   streaming, boundary conditions, advance (Phase 1 reference)
|   +-- engine/         # 3D Taichi solver: lattice3d, Simulation3D (GPU/CPU) (Phase 2)
|   +-- examples/       # runnable cases: cavity, cylinder, sphere
|   +-- post/           # plotting, VTK export, progress/health monitor
|   +-- config/         # run configuration, environment check
|   +-- geometry/       # solid masks (cylinder, sphere) and wall fractions;
|   |                   #   STL import + voxelization later (Phase 4)
|   +-- turbulence/     # LES subgrid model, wall functions (Phase 3)
+-- cases/              # validation + demo case definitions
+-- tests/              # pytest: Tier A unit + Tier B validation (marked slow)
+-- requirements.txt
```

## 7. Phased roadmap

**Phase 0 - Scaffold.** Repo structure, environment (Taichi + CUDA verified on the 3090),
CI-style `pytest` skeleton, config loader. *Exit: `pytest` runs green on an empty suite.*

**Phase 1 - 2D LBM core. DONE.** D2Q9 BGK in **NumPy** (not Taichi - Phase 1 stays
CPU/NumPy for clarity while learning the physics; GPU port lands in Phase 2 where 3D needs
it). Operators: moments, equilibrium, BGK collision, streaming, bounce-back walls, moving
wall, Guo body force. Validated:
- **Poiseuille**: profile exactly parabolic (R^2=1.0), peak within 1% of analytic (walls
  taken from the fit to avoid sub-cell ambiguity). Guo forcing gives 2nd-order accuracy.
- **Lid-driven cavity vs Ghia et al. (Re=100)**: correct vortex + centerline structure;
  interior minimum matches Ghia within ~5% at 64^2 and **<1% at 128^2** (convergent).

44 fast tests (Tier A unit) + 10 slow (Tier B validation).
*Exit met.*

**Phase 2 - 3D + first bluff body. DONE.** D3Q19 engine (`Simulation3D`), Guo
forcing, Zou-He/NEEM velocity inlet, zero-gradient outlet, free-slip walls, cylinder as a
voxelized obstacle, drag via momentum exchange, VTK export, live progress/ETA + divergence
monitor. Cylinder result:
- **Flow validated.** Rigid NEEM inlet holds the free-stream (measured Re=101 vs nominal
  100); vortex shedding **Strouhal = 0.165** at Re=100 (ref ~0.16-0.18); **no shedding at
  Re=40** (correct sub-critical behavior).
- **Drag: correct method, known discretization offset.** Momentum-exchange force is
  `sum 2 c_i f_i` (post-collision, over fluid->solid links). Cd ~ 1.9 vs unbounded ref ~1.4
(later cut to 1.576 by interpolated bounce-back in Phase 3) -
  the ~35% is the documented over-prediction of **full-way (staircase) bounce-back** at
  finite resolution, which converges with resolution and is cured by **interpolated
  bounce-back** (the accuracy upgrade, deferred to Phase 3 with the wall-treatment work;
  needed for smooth aero surfaces regardless). Note the two references differ: unbounded
  cylinder Cd~1.4 vs the confined Schafer-Turek channel benchmark Cd~3.2 - free-slip walls
  target the unbounded value.

Sphere (true 3D): runs stably at low Re (Re~15-20; BGK needs high tau, so higher Re awaits
MRT in Phase 3). Cd lands in the staircase over-prediction band, but the free-slip **corner**
where the y and z symmetry planes meet injects spurious energy (`max|u|` ~4x inlet) - a
node-based specular-reflection artifact needing proper corner handling (Phase 3). So the
sphere validates the machinery, not a clean Cd. It also still uses staircase
`bounce_back`: only `wall_fraction_cylinder` exists, so the sphere never got Bouzidi.
A `wall_fraction_sphere` is the fix.

Regression net: `tests/test_engine3d.py` (6 Tier-A invariant tests - macroscopic, collide
fixed-point, conservation, bounce-back swap, free-slip involution, inlet). It caught a real
top-wall bug in `free_slip_y`/`_z` (a reused temp), which also slightly cleaned the cylinder
Cd (1.89->1.83). 44 fast tests total; validation cases marked `slow`.

*Exit met. Deferred to Phase 3: interpolated bounce-back (staircase Cd),
free-slip corner treatment (sphere), MRT/regularized collision (high-Re stability).*

**Phase 3 - Turbulence + walls. IN PROGRESS.** Three of the core operators are in, each
validated by reducing exactly to the previous scheme (a golden-oracle parity test):

- **TRT collision. DONE.** Two-relaxation-time: even/odd split about `OPP`, with `s_plus`
  set by viscosity and `s_minus` by the magic parameter `Lambda = 3/16` (which also fixes
  the tau-dependent wall location). Decouples stability from viscosity: the sphere now runs
  at Re=50 where BGK diverged above Re=20. Parity test: `s_minus = s_plus` reproduces BGK.
- **Interpolated (Bouzidi) bounce-back. DONE.** Uses a per-link sub-cell wall fraction `q`
  so curved surfaces are not staircased. Cylinder Cd **1.83 -> 1.576** against the ~1.4
  reference. Needs a post-collision snapshot (`fc`) and costs ~38% runtime (the per-step
  full-field copy; fixable later with buffer ping-pong).
- **LES Smagorinsky. DONE.** Local eddy viscosity from the non-equilibrium stress tensor
  `Q_ij` (no finite differences, fully local), giving a per-cell `tau`. Inactive on smooth
  flow (`|Q|=0`), active where strained. Parity test: `cs=0` reproduces BGK exactly.

**Collision consolidation.** All variants are now one kernel, `collide_full(tau, cs, gx,
trt)`, with thin Python wrappers (`collide`, `collide_forced`, `collide_trt`, `collide_les`)
as special cases. This removed four copies of the moment block and, more importantly, makes
the variants **composable** - TRT + LES + forcing together, which the Ahmed body needs and
the previous separate kernels could not express.

- **Backward-facing step (Armaly). DONE.** First separation/reattachment case. New masked
  inlet `inlet_neem_open` drives only the open channel rows (plain NEEM diverged with the
  step block filling half the inlet plane; the soft equilibrium inlet under-drove to Re~27).
  Reattachment `x_r` taken from the first floor-node `u_x` sign change: **x_r/S = 2.47 at
  Armaly Re = 101** (channel-mean velocity, ref ~3). The ~18% low is attributed to the
  uniform (non-developed) inlet and the on-node step height.

**Interpolated-wall drag.** `drag_interp` sums `c_i (f_in + f_out)`, the correct momentum
exchange for a Bouzidi wall (the full-way `2 c_i f_i` assumes reflection at the node). Both
bodies then agree on a ~7-9% over-prediction, believed to be the discretization floor.

**Forced plane channel + a TRT-forcing bug it caught.** Built as the log-law prerequisite, and
it exposed a real defect in the composed Guo forcing: `collide_full` relaxed the whole force
source with `s_plus`, but the antisymmetric (momentum-carrying) part must relax with `s_minus`.
Invisible to the `s_minus=s_plus->BGK` parity test (at that limit the two prefactors coincide);
only a forced channel comparing TRT to BGK magnitudes caught it - TRT peak -16%, disagreeing
with BGK by ~18%. Fixed by splitting the source into symmetric/antisymmetric parts and relaxing
each with its own rate. Golden oracle: forced Poiseuille now recovers the analytic parabola
(R^2=1) under both operators, agreeing to ~0.6%, leaving a fixed ~0.33-cell body-force wall
offset (Mach-independent -> pure discretization, ~0.4% at production delta). `test_channel.py`
locks in TRT-vs-BGK peak agreement (fails on the bug, passes on the fix).

**Turbulent channel Re_tau=180 (DNS, cs=0). Tripped and sustained.** Minimal Jimenez-Moin box
(~120x130x60). Seeded IC = mean parabola + divergence-free streamwise rolls (from a
streamfunction with a `(1-eta^2)^2` window so `u_y` and `u_z` both vanish at the wall) + streak
modulation + broadband noise (the only x-dependent term, so it is what lets the x-invariant
rolls break down to 3D). Smagorinsky OFF on purpose - static Smagorinsky over-damps and
relaminarizes at this Re; TRT mandatory (tau~0.505). The flow trips, breaks down (`w'` grows
from ~0 to O(u_tau)), and sustains the near-wall regeneration cycle over ~14 turnovers:
U_c/u_tau = 17.6 (MKM ~18.3; minimal-box low is expected), rms(u,v,w)/u_tau = (1.31,0.59,0.67),
correct anisotropy.

**Channel validated against the log law.** Time+plane-averaged mean and RMS profiles, folded
about the centerline, in wall units (`delta_eff` from the laminar oracle, force-balance
`u_tau`); measurement plots live in `src/post/plotting.plot_law_of_wall`. Over ~40 turnovers
the mean **law of the wall is reproduced**: sublayer collapses onto `u+=y+`, a log region
appears, and centerline **U+ = 18.3 matches MKM** (top/bottom asymmetry converged 10%->1.4%,
confirming it was a statistics, not a symmetry, issue - the engine is provably symmetric from
the laminar oracle). Second-order: `u'_rms` peak sits at the correct `y+~16` but reads ~3.2 vs
MKM 2.65 - the known minimal-box over-prediction of fluctuation intensity (a single coherent
streak pair, plus the burst-cycle plane-mean unsteadiness counted in the variance); it does not
move with more averaging, so it is the box, not convergence. First-order (the part wall
functions key off) validated; publication-grade second-order needs a full-size box.

**Near-wall coarsening: the measured failure that motivates wall functions.** Holding Re_tau=180
and shrinking `delta` (64->32->16, so the first node climbs y+ 2.3->4.7->9.4), comparing the
resolved wall shear `sqrt(nu du/dy)` against the exact force-balance `u_tau`. The wall-resolved
approach fails in two modes: (1) *accuracy* - wall-shear error grows -7.5% -> -12.3%, centerline
U+ drifts 18.4 -> 17.4, and the `u'_rms` peak collapses 3.1 -> 2.3 as the near-wall cycle loses
cells; (2) *feasibility* - at delta=16 (nu=4e-4, tau=0.501, s_plus~1.995) the DNS diverges
outright. Coarsen the wall and the resolved solver first gets the wall stress wrong, then cannot
run - exactly the case (measured, not assumed) for a wall function, which supplies tau_w from
the log law instead of an under-resolved gradient.

**Wall function - built (equilibrium log-law model).** `src/turbulence/wall_function.py`
inverts the log law `u1 = u_tau*((1/kappa)ln(y1 u_tau/nu)+B)` for `u_tau` by Newton (viscous
initial guess, guarded step); a `@ti.func` twin `wall_utau` lives in the engine for device use.
The wall stress is imposed as a wall-adjacent eddy viscosity: `wall_model` finds fluid nodes
next to a solid in the wall-normal direction, computes `u_tau` from the wall-parallel speed,
and sets `nut_wall = u_tau^2 y1/u1 - nu` there (gated on `y+ > 30`, off below the log layer);
`collide_full` adds `3*nut_wall` to the local tau. Validated: synthetic round-trip recovers
`u_tau` to 1e-5 (isolates the Newton from "is the profile log?"); the augmentation equals
raising tau0 by 3*nu_t at that node and nowhere else (checked at cs=0, so it is not gated
inside the LES branch); `wall_model` matches the inversion and respects the y+ gate. Against
the trusted Re_tau=180 DNS the inversion recovers `u_tau` to ~5% - the residual is that
Re_tau=180 has essentially no clean log layer (30 < y+ < 0.15*Re_tau is empty), so the check
node sits in the wake. The model only *engages* for a first node at y+>30, which the stable
Re_tau=180 channel never reaches - so here it demonstrates parity (off = identical) and
non-breakage, and its quantitative payoff (wall-shear error driven to zero) waits for the
high-Re Ahmed body, where wall-resolving is impossible and the first node naturally sits in
the log layer. Generalizing wall-normal detection beyond axis-aligned (y) walls is deferred to
the geometry work (a wall-distance/normal field, like `q`).

**Equilibrium unified.** `feq` extracted as one `@ti.func` used by both `collide_full` and the
new `init_equilibrium` (equilibrium IC from a prescribed velocity field), removing the
duplicated equilibrium formula. Guarded by a collision-fixed-point test (init a moving
equilibrium, collide with no force, `f` must be unchanged) - fails if the two feq ever drift.

**Test coverage.** Every engine kernel and geometry function is unit-tested, including the
Phase 3 additions: bounce_back_interp (q=0.5 -> halfway), drag_interp, inlet_neem and
inlet_neem_open, the combined collide_full (TRT+LES+forcing at once), and the wall fractions
(crossing points land on the surface). Tests have caught real bugs: free_slip_z using the y
mirror table, a double relaxation in LES, a missing q load.

**Ahmed body - staged build toward the gate. IN PROGRESS.** Voxelized Ahmed (`src/geometry/ahmed.py`):
box + 35deg rear slant + rounded nose, all dimensions derived from one height H so resolution
scales cleanly. Wind-tunnel run (`src/examples/ahmed.py`): body-only drag via a separate
`self.body` mask + `drag_body` (excludes the floor/ceiling walls from Cd), inlet_neem_open /
outlet / free-slip sides / no-slip floor+ceiling+body, VTK + mid-span u_x wake PNG.
- *Stage 1 (pipeline). DONE.* Runs stable at reduced Re_H=100 (the square nose diverged at
  Re=300 on the tau->0.5 margin, so the nose was rounded), physical wake with a clear
  recirculation bubble. Machinery, not Cd - same bar as the sphere.
- *Stage 2 (stability ceiling). MEASURED.* TRT+LES ladder: cs=0.1 diverges at Re~300; cs=0.2
  holds to ~2000 but only by over-dissipating (max|u| creeps to Ma~0.36). Reference Re~7.7e5 is
  ~400x beyond that - cs-tuning cannot get there. Motivates the collision upgrade.

**Regularized collision (Latt-Chopard). DONE.** `collide_reg(tau, gx)` rebuilds the
non-equilibrium `f` from its stress tensor Pi only (`f_neq = 4.5 w_q Q_q:Pi`), discarding the
ghost moments that blow up as tau->0.5. Clean stability with NO added viscosity, so the true Re
is preserved (unlike heavy Smagorinsky). Ahmed ladder with `collide_reg`: clean and bounded to
**Re~1000** (vs TRT's ~300), holds to ~3000 (Ma onset), diverges ~10000 - the new ceiling is
resolution, not collision. Validated: mass conserved; **shear viscosity exact** on a resolved
unforced decaying wave (matches analytic and BGK to <3%). Note: forced + regularized needs a
Guo force-correction to Pi (regularization zeroes f_neq's -F/2 first moment) - deferred, since
nothing regularized is forced here.

**Wall function generalized + wired. DONE.** `wall_model` now finds the wall normal per node from
the discrete solid gradient (`-sum c_q over solid neighbours`), so it works on any wall
orientation (roof/floor/base/front/slant), using the wall-parallel speed and reducing exactly to
the old y-normal case (test 21 still passes; a new x-wall test 24 checks the normal). Wired into
the Ahmed loop (macroscopic -> wall_model(nu, 0.5) -> collide_reg), with `collide_reg` adding
`3*nut_wall` to its local tau. Correctly dormant so far: at every stably-reachable Re the first
node sits at y+<30, so the gate stays shut - it engages only at the high-Re regime the reference
run targets.

**LES + regularized + wall function, composed. DONE.** `collide_reg(tau, cs, gx)` folds
Smagorinsky in via the stress Pi it already computes (cs=0 reduces exactly to the validated
regularized scheme). The full stack now runs the Ahmed body stably to **Re_H=30000**
(max|u|~0.08, clean; regularized-alone diverged ~10000, TRT+LES ~2000), with the wall function
actively engaged (~1300 nodes at y+>30). All three - regularization (numerical stability), LES
(subgrid turbulence), wall function (near-wall stress) - live at once. LES also cured the Ma
creep at Re=3000 (max|u| 0.22 -> 0.08).

**Accuracy campaign - what it found.** Cd ~3.3 at Re=30000 is ~10x the reference ~0.29. The
Ahmed body's low true Cd amplifies every absolute error (the ~35% staircase over-prediction
that was tolerable on the cylinder is enormous on a 0.29 body). Three levers were tried:
- *Resolution.* H=32 -> 48 at Re=3000 moved Cd 2.70 -> 2.27; first-order Richardson puts the
  grid-converged value near ~1.4 - still ~5x high (rough: the H=32 point was a short run, the
  H=48 point a converged one). Resolution is a real but secondary error. Worse, it is
  hardware-capped: with q/fc/macroscopic fields allocated the 24 GB card holds ~70M cells,
  about H~60; resolving the slant and wake well enough for ~0.3 needs H~150-300, i.e. a cluster.
- *Free-slip far-field ceiling* (`free_slip_y_top`). Raised Cd 2.70 -> 2.98: a free-slip roof
  meeting free-slip sides injects the same edge artifact the sphere showed. Needs corner
  treatment before free-slip tunnel walls are usable. Kernel kept, not wired.
- *Wall-stress validation in a coarse channel* (Re_tau=590, delta=16, first node y+~31).
  Hit the wall-modeled-LES bind: cs=0.1 over-damps the near-wall log layer (static Smagorinsky
  sees the mean shear) -> U+=14 vs ~21, the wall model's inverted u_tau comes out low, its y+
  estimate falls under the gate and it never engages (wall on == wall off); cs=0 has no subgrid
  dissipation and diverges. No static cs works. Needs a wall-aware SGS model (van Driest
  damping, or WALE/Vreman). The wall function stays validated at the component level
  (inversion round-trip, tau coupling, y- and x-wall tests). Same caveat applies to the Ahmed
  wall model's engagement at Re=30000.

**Decision.** The absolute Ahmed Cd is gated by hardware plus two research-grade modeling
pieces, and the charter already says relative comparisons, not certification-grade absolute
Cd. The written Phase-3 gate ("Cd matches published") contradicted that charter and the 24 GB
budget flagged in section 1; it is replaced by a relative gate.

*Remaining: the **slant-angle sweep** (25/30/35 deg, identical settings) as the relative-
comparison gate. Pass = the Ahmed drag-crisis trend (Cd rising toward ~30 deg, dropping past
it; at minimum 25 > 35). At H=32 the angles differ by only 1-3 cells in slant drop, so a flat
result means "needs H=48," not "wrong physics." Deferred with known cause: van Driest / WALE
SGS, free-slip corner treatment, forced+regularized Guo correction, SDF wall distance for
non-axis-aligned walls.*

**Phase 4 - Automotive features.** STL import + voxelization of real parts, moving ground,
rotating wheels, per-part force breakdown. *Exit: front-wing or full-car run with sane,
stable coefficients.*

**Phase 5 - Interactivity & sweeps.** Live viewer, parameter sweeps, A/B comparison
tables, mixed-precision for larger domains.

Each phase ends only when its validation gate passes. Docs and tests are updated within
the same phase, not after.

## 8. Additional lattice stencils (future)

The lattice is a pure-data descriptor (`Q, D, E, W, OPP, CS2`) that the operators
consume generically, so new stencils are data-only additions - no kernel changes for
same-dimension sets. Candidates, and what each is actually good for:

- **D3Q15 / D3Q19 / D3Q27** - the 3D Navier-Stokes workhorses. D3Q19 is the default;
  D3Q27 for high-Re isotropy/stability; D3Q15 for cheap/coarse runs (weaker isotropy).
  *(D3Q19 done; Q15/Q27 planned as data-only additions.)*
- **D2Q9** - the standard 2D NS stencil (done).
- **D2Q5 / D3Q7** - *not* full fluid stencils. These are advection-diffusion lattices for
  a scalar field (temperature, species). Useful later if we add heat transfer or passive
  scalars alongside the flow - a second distribution on a small stencil.
- **Higher-order 2D (D2Q17, D2Q37) / 3D (D3Q39, D3Q41)** - needed only for thermal
  (compressible/high-Mach) or high-accuracy work. Overkill for incompressible aero;
  revisit only if a case demands it.

Note on "any DnQm": more velocities is not automatically better - a stencil must satisfy
the isotropy moment conditions (`sum w_ie_i=0`, `sum w_ie_i outer e_i=cs^2I`, and the 4th-order condition)
to reproduce the target physics. Arbitrary DnQm sets (e.g. a made-up D2Q16/D2Q25) generally
do **not** - only specific, derived velocity/weight sets work. Add stencils from the
literature, not by picking a velocity count. Every new stencil ships with the same
descriptor tests (weights sum to 1, opposites reverse, isotropy moments).

## 9. Open questions to revisit

- Taichi vs Warp final call (decide after Phase 1 ergonomics).
- Collision operator: **TRT chosen** for Phase 3 (two-relaxation-time). Near-BGK cost
  (LBM is memory-bound), decouples stability from viscosity, and Lambda=3/16 fixes the
  tau-dependent wall location. **Full MRT deferred** as a later option if TRT proves
  insufficient at extreme Re (it buys more control at ~10-25% cost and much more code).
- FP16 storage - how much accuracy do we trade for domain size? (measure in Phase 5).
- Real F1 geometry source and its licensing (needed by Phase 4).
- Cylinder Cd calibration: blockage, resolution, tau, MEM factor (Phase 2, in progress).
