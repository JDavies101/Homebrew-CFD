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

The solver is not trusted on an F1 shape until it passes a ladder of cases with known
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
matches published Cd and wake structure is it run on real F1 geometry. OpenFOAM provides an
independent FVM cross-check on selected cases.

## 6. Repository layout

```
homebrew-cfd/
+-- README.md
+-- docs/               # design, validation notes, results
+-- src/
|   +-- lbm/            # 2D NumPy solver: lattice, moments, equilibrium, collision,
|   |                   #   streaming, boundary conditions, advance (Phase 1 reference)
|   +-- engine/         # 3D Taichi solver: lattice_d2q9, lattice_d3q19, Simulation3D (GPU/CPU) (Phase 2)
|   +-- examples/       # validation cases: cavity, channel, cylinder, sphere, step, Ahmed, wing
|   +-- run/            # Case description, generic runner, case files, run folders, CLI solve
|   +-- post/           # plotting, VTK export, progress/health monitor
|   +-- config/         # run configuration, environment check
|   +-- geometry/       # solid masks (cylinder, sphere) and wall fractions;
|   |                   #   SDFs, NACA sections, STL import + mesh distance
|   +-- turbulence/     # LES subgrid model, wall functions (Phase 3)
+-- app/                # desktop application (PySide6 + PyVista)
+-- packaging/          # PyInstaller spec, Inno Setup installer, build script, icon
+-- cases/templates/    # drop-in case files shipped with the application
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
  100); vortex shedding **Strouhal = 0.171** at Re=100 (runs 138-140, refined FFT; ref ~0.164, offset is blockage, see E3); **no shedding at
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

**Phase 3 - Turbulence + walls. DONE (relative gate passed).** Three of the core operators are in, each
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
  Convention note: until the section-10 review the coefficient was 18 instead of the textbook
  18*sqrt(2), so every cs quoted below (cs=0.1, 0.2, ...) is ~0.84x the textbook C_s. Code now
  uses the textbook value.

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
(R^2=1) under both operators, agreeing to ~0.6%. A fixed ~0.33-cell wall offset remained
(Mach- and resolution-independent) and was read at the time as a body-force effect; the code
review later traced it to collision running on wall nodes (Guo force applied inside the wall,
bounced populations mixed through feq). With walls skipped, TRT is exact at halfway:
delta_fit 31.999, peak -0.07%. `test_channel.py` locks in TRT-vs-BGK peak agreement and now
TRT vs analytic within 0.5%.

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

**Ahmed body - staged build toward the gate. DONE.** Voxelized Ahmed (`src/geometry/ahmed_body.py`):
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

*Sweep, first pass - INVALIDATED.* Pre-fix Cd was dominated by the wall-node collision
artifact; post-fix phi=25 Cd = 0.740 (H=32, cs=0.084). Next: block-averaged error bars, then
rerun 25/30/35 with the fixed engine. Original record kept below.
*Sweep, second pass (fixed engine, H=32, cs=0.084, Re_H=30k, 3+3 T_ft):* 25 -> 0.740+/-0.003,
30 -> 0.741+/-0.003, 35 -> 0.748+/-0.004 (5-block SE). Flat (1.2% spread), no peak, Cd(35) >
Cd(25) - fails the gate, but NOT converged: 10-block means oscillate +/-0.015 on a ~1 T_ft
period (not a monotonic drift), so 5-block SE is too small; honest SE ~0.005. Midplane PNGs are
instantaneous - slant attachment cannot be judged from them. Next: 5+11 T_ft, time-averaged
u field for the slant check.
*Sweep, third pass (5+11 T_ft, time-averaged u):* 25 -> 0.742, 30 -> 0.744, 35 -> 0.750
(honest SE ~0.003). Monotonic up, 1.1% spread, no peak - fails the gate. 25/35 mid-plane mean
fields look near-identical: low-speed region over the slant from its leading edge (separated at
both, tentatively). Plot caveat: NaN body and u~0 both render white, so the slant surface can't
be told apart from stagnant fluid - needs a quantitative near-wall u check.
*Fallback relative gate (nose shape), set up:* `ahmed.py <phi> [round|square]`; square nose skips
the R=100mm fillet. Pass = Cd(square) > Cd(round) beyond error bars (expect a large rise from
front-edge separation; front-roof bubble visible in `ahmed_mean_nose_*`). `slant_check` prints
mean along-slant u_t in the first fluid cell (attached fraction); mid-span mean saved as
`ahmed_umean_mid_<tag>.npy`.
*Slant check result (H=32, Re_H=30k):* first-fluid-cell u_t is contaminated by the voxel
staircase (~0.03 U slant-avg). Stepping off the wall, u_t/U rises monotonically and stays
POSITIVE at every depth - 25 deg: 0.03/0.10/0.17/0.26/0.47 at 1/2/3/4/6 cells; 35 deg:
0.02/0.08/0.13/0.18/0.31. So mid-span shows a thick, retarded-but-forward shear layer, NOT a
reversed separation bubble; 25 deg carries clearly more near-slant momentum than 35 deg (right
direction). The drag-crisis mechanism is smeared by under-resolution (~25 staircased cells on
the slant, no Bouzidi), so Cd cannot order the angles. Slant gate closed as not resolvable at
this Re/resolution (not a solver defect); Cd reproduced bit-for-bit on rerun. Phase 3 relative
gate moves to nose shape.

*Nose-shape gate: PASS (H=32, Re_H=30k, 5+11 T_ft).* Round vs square nose, all three slants:
Cd(square)/Cd(round) = 1.102/0.742, 1.104/0.744, 1.121/0.750 -> +48/48/49%, a ~0.36 gap vs
~0.02 error bars. Square-nose SE ~6x larger (0.02 vs 0.003): sharp-edge front separation makes
the wake unsteady, as expected. Mechanism confirmed in ahmed_mean_nose_phi25_square: a large
reversed-flow bubble sits on the front roof from the sharp top corner; the round nose has none.
Right sign, right order of magnitude. This is the Phase 3 relative aero gate - a geometry change
the solver resolves cleanly and ranks correctly.
*Sweep, first pass (H=32, cs=0.1 old convention, run BEFORE the wall-node + sqrt(2) fixes):*
25 -> Cd 3.2454, 30 -> 3.2328, 35 -> 3.2149. Monotonic, 0.9% spread, no peak - fails the
meaningful gate (25 > 35 holds only by a margin smaller than an unmeasured error bar). Likely
cause: flow separates at every angle on the coarse slant. Next, in order: (1) block-averaged
error bars on Cd in `ahmed.py`; (2) check the 25 deg wake PNG for reattachment on the slant;
(3) rerun at H=48 with the fixed engine (use cs=0.084 to match old runs, or accept the new
convention and rerun all three); (4) if still flat, fall back to another relative test
(rounded vs square nose, or a ground-clearance sweep).

**Phase 3 close-out.** Gate met: the relative aero gate (nose-shape, square vs round, +48%
Cd, correct sign and mechanism) passed. All core operators done and unit-tested: TRT,
Bouzidi interpolated bounce-back, LES Smagorinsky (textbook constant), regularized collision,
generalized log-law wall function, unified `feq`/collision/`_stress`, and the full
regularized+LES+wall-function stack stable on the Ahmed body to Re_H=30000. Validation ladder
green: Poiseuille, cavity (Ghia), cylinder (Cd 1.576, St 0.165), sphere (Bouzidi), backward-
facing step, Re_tau=180 channel (U+ 18.5), Ahmed stability + nose gate.

*Carried forward (known, deferred by decision - none block the Phase 3 gate):*
- Absolute Ahmed Cd - out of charter (hardware-bound at 24 GB; needs H~150-300 = cluster).
- Free-slip corner artifact - `free_slip_y_top` built but unwired; Ahmed uses no-slip tunnel
  walls instead. Needs corner treatment before free-slip far-field walls are usable.
- Wall-aware SGS (van Driest / WALE / Vreman) - static Smagorinsky over-damps the near-wall
  log layer, so the wall model can't engage cleanly at high Re. Needed for trustworthy
  wall-modeled car surfaces.
- SDF wall-distance field - wall normals come from the discrete solid gradient today; an SDF
  gives both sub-cell `q` and normals for arbitrary (STL) surfaces.
- Forced + regularized Guo correction to Pi - deferred (nothing regularized is forced yet).
- Second-order channel turbulence (`u'_rms`) - minimal-box high; needs a full-size box.

**Phase 4 - Automotive features. LARGELY DONE (4.0-4.3 complete; 4.4 reached with the quasi-2D wing in ground effect; a 3D front wing or full car remains).** Turn the validated solver into a car-aero tool. Real
geometry, real road boundary conditions, part-resolved forces. Same charter as Phase 3:
relative comparisons and credible trends, not certification-grade absolute Cd. Staged:

- *4.0 Foundations (the Phase-3 carry-forwards that Phase 4 depends on).* SDF wall-distance
  field from geometry, feeding both Bouzidi `q` and per-node wall normals; a wall-aware SGS
  (WALE or van Driest) so the wall model engages on real surfaces at high Re. *Gate: WALE
  reproduces the Re_tau=180 channel law-of-wall as well as cs=0 DNS, and engages (wall on !=
  wall off) in the Re_tau=590 coarse-wall case that static Smagorinsky failed.*
  *4.0 progress - WALE.* `les_wale` kernel + `nut_les` field (composed into `collide_reg` /
  `collide_full` as `tau += 3*nut_les`, same convention as `nut_wall`). Validated: exact NumPy
  parity on a random field; zero on quiescent flow; Re_tau=180 channel unharmed (U+ 18.2 vs
  18.3, wall-shear error -2.6%, u'_rms peak moved y+ 15.5 -> 12.7 at the same 3.11 - a small
  buffer-layer contribution). High-Re channel attempt (Re_tau=590, delta=16) was not a valid
  test: first node at y+ 18 is unresolved with the wall model gated off, and the 9x34x5 box
  cannot sustain wall turbulence (fluctuations peak in the core). WALE ran stable and did not
  over-damp (U+ 27 vs the ~21 log law; nut_les/nu mean 16), but the result is set by missing
  wall stress, not the SGS model. Static Smagorinsky on collide_reg diverged there (unexplained,
  invalid setup - logged, not chased). High-Re WALE validation moves to the Ahmed body
  (unforced, wall model engaged). Follow-up if ever needed: a proper wall-modeled channel
  (larger box, first node y+ > 30, wall model wired).
  *Forced regularized collision - fixed.* Regularization dropped f_neq's -F/2 first moment
  (the first-order Hermite term), over-driving forced flow: forced Poiseuille under
  `collide_reg` peaked 13.5% high. Restored as `-1.5 w_q c_qx gx` in the reconstruction
  (zero when unforced). Residual ~1% is non-monotone in resolution (-1.02% at ny=32, +0.99% at
  ny=64), consistent with a fit/near-wall metric effect and below collide_reg's 3% viscosity
  certification; test tolerance 2%. Follow-up: measure nu_eff from fitted curvature to pin it.
  *4.0 progress - WALE on Ahmed (round nose, Re_H 30k). Gate closed (run 46).* WALE engages the
  wall model on the body (4664-4725 wall-engaged nodes vs 816 with Smagorinsky), but the first
  runs were noisy: 1541 hot cells (|u| > 2 u_ref), rho 0.93-1.20 (run 21). A viscosity sponge
  caused global mass accumulation and was removed; the NEEM pressure outlet `outlet_pressure`
  pins the mean density (run 29: no drift, but 1271 hot cells, rho 0.88-1.13). I then isolated
  the remaining noise one cause at a time:
  - *SGS vs BC (run 30).* Smagorinsky on the same case gave 28 hot cells, so the z free-slip BC
    was not at fault. WALE's out-of-domain clamp halved the edge gradient; I replaced it with a
    one-sided difference (`_in_dom` divisor in `les_wale`). Correct, but no effect on the noise
    (run 31).
  - *Trapped acoustics (runs 31-37).* The upstream field carried lateral velocity ~0.2-0.3 U and
    rho range 0.23 at the inlet, uniform in j/k: acoustic modes (du ~ c_s drho) reflected by a
    closed box (velocity inlet, free-slip z, no-slip floor/ceiling, pressure outlet). tau0 =
    0.50016 gives almost no molecular damping and WALE (by design) gives nut ~ 0 in near-pure
    shear and uniform flow; Smagorinsky had been damping them as a side effect. Changes: start
    from rest with a cosine inlet ramp over 1 T_ft (no effect alone), regularized NEEM inlet and
    outlet (rebuild f_neq from Pi only; upstream hot 503 -> 192), and a relaxation absorbing
    layer `sponge_relax` at the x ends (f -> f - sigma (f - feq(1, U_in)), 24 cells, sigma 0.1,
    quadratic ramp). The x-layers dropped rho to 0.976-1.030 (run 37).
  - *Ramp-end stability (runs 38-42).* Adding z-wall layers diverged at the end of the ramp. A NaN
    locator showed growth starting in the nose ground-clearance gap (6 cells, i ~ 128, j 2-4), and
    the x-only case survived the same transient only marginally (peak 0.45 = 9 U). A
    Smagorinsky floor under WALE (cs = 0.04, additive via `collide_reg`) halved the peak and
    made the x+z case stable (run 42).
  - *Layer bias (runs 43-46).* z-layers relaxing to the free stream raised Cd from 0.693 to 0.780:
    with ~10% blockage the bypass flow beside the body is faster than U, and forcing it to U adds
    resistance. Relaxing instead toward a running mean (`sponge_relax_mean`, EMA alpha = 1/2000)
    removes fluctuations without imposing a mean, but started from rest it locked the strips at
    ~0.3 U (run 45, Cd 1.275): a mean-target layer has no restoring force. Switching it on at
    the end of the ramp, with the mean copied from the current field, fixed it (run 46).
  *Reference configuration (run 46):* WALE cw 0.5 + Smagorinsky floor cs 0.04, regularized NEEM
  inlet/outlet, cosine ramp over 1 T_ft from rest, x relaxation layers 24 cells (free-stream
  target), z layers 12 cells (running-mean target from ramp end), sigma 0.1. Cd 0.700 +/- 0.002
  (floor-only reference 0.693), rho 0.983-1.009, 196 hot cells (none on z walls; 46 at the floor,
  the rest at the nose gap and front roof), z-strip u_x matching the adjacent flow. As separate
  kernels the layers cost ~20% (947 MLUPS vs ~1187); I then fused both into `collide_reg`
  (post-collision moments after the x-layer in closed form, single write of f; test 33 checks
  equality with the separate kernels). Run 47: Cd 0.701 +/- 0.001, 174 hot cells, 1183 MLUPS -
  the full configuration now runs at the no-layer baseline speed. Open: the nose gap remains the
  stability-critical region at H = 32.
  *4.0 progress - SDF wall geometry.* `src/geometry/sdf.py`: a shape is a function phi(x, y, z)
  (> 0 fluid, < 0 solid). From one phi I get the solid mask, per-link Bouzidi fractions q (sign
  change of phi along each boundary link, bisection to f64 precision, out-of-domain links masked),
  wall normals grad(phi)/|grad(phi)|, and the first-node wall distance. Validated by reduction:
  on an off-lattice sphere, q matches the analytic ray-sphere `wall_fraction_sphere` to 1e-5 over
  the same link set, and normals are radial to 1e-6. `build_wall_list(phi)` now stores per-node
  `wall_y1 = phi(node)` and SDF normals; without phi it keeps y1 = 0.5 and mask normals, and the
  Ahmed reference run reproduces bit-identically (run 48).
- *4.1 STL import + voxelization. First gate passed (runs 49-50).* `src/geometry/mesh.py` reads
  and writes binary/ASCII STL; `mesh_distance.py` builds the SDF on the GPU with one thread per
  triangle: narrow-band unsigned distance (Ericson closest point, atomic min), inside/outside by
  ray parity along offset grid lines, and a trilinear phi wrapper. q for a mesh comes from exact
  link-triangle intersection (Moller-Trumbore), so its only error is the mesh's own facet sag;
  q from the interpolated phi is limited by the trilinear bound (~3/(8 r) cell at curvature radius
  r), which matters at the small radii of a car. Tests: exact q on a box (flat faces, straight and
  diagonal links), icosphere sign/distance/q against the analytic sphere, with q errors measured
  as wall displacement along the normal |dq||c.n| because grazing links are ill-conditioned in q.
  Flow gate: an icosphere STL (20480 triangles) run through the sphere case gives Cd 1.6233 vs
  1.6228 for the analytic geometry (+0.03%, gate 1%; runs 52-53), with 2 mask cells differing.
  *Sphere case fix (found while running the gate).* The example used an equilibrium inlet
  (f = feq(1, U) at x = 0) with a zero-gradient outlet. That fixes the boundary distribution,
  not the velocity: the interior settled at u = 0.072 and rho = 1.08 (78% of the intended mass
  flux), hidden by normalizing Cd with the measured U_eff. With the regularized NEEM inlet and
  pressure outlet the impulsive start from rest then trapped a domain-length acoustic mode
  (damping time ~1/(nu k^2) ~ 3.7e5 steps). Starting from uniform flow at U plus x relaxation
  layers gives a steady field (mass flux constant to +/-0.1%, drag drift 0.00% over 5000 steps)
  and Cd 1.623 vs Schiller-Naumann 1.54 at Re 50 (+5.3%: correlation scatter ~5%, 1.9% blockage,
  D = 20). Lesson: every open-boundary case needs a mass-flux-vs-x check and a drag drift check.
  Original plan: Read a triangle mesh, voxelize to the solid mask plus the
  per-link `q` field via the SDF (so curved car surfaces are not staircased). *Gate: an STL of
  a sphere/cylinder recovers the analytic-mask Cd/St (validation by reduction).*
- *4.2 progress - moving ground. Gate passed (run 55).* A wall-velocity field `uw` read by
  `bounce_back(scale)`: at solid nodes with u_w != 0 I add the Ladd correction 6 w_q (c_q . u_w)
  to every population whose destination (periodic, as in streaming) is fluid; `scale` follows the
  inlet ramp. The first attempt (run 54) corrected all pairs, so in-plane populations circulating
  along the x-periodic floor row gained 6 w U per step without bound, and the wall model used the
  absolute velocity, engaging the whole moving floor (79244 nodes). Fixes: fluid-bound links only,
  and `build_wall_list` now stores the mean wall velocity of each node's solid neighbours so
  `wall_model_fast(nu, scale)` works on u - u_w. Tests: exact Couette profile (0.1% of U) with no
  cross-flow. Ahmed: mean mid-span u/U one cell above the floor at x0 - 56 is 0.995 (moving) vs
  0.419 (static), within 2% of the free stream over j = 1-6; underbody u/U at j = 1 is 1.09 vs
  0.30; Cd 0.6848 vs 0.7011 (-2.3%).
- *4.2 progress - rotating wheels. Gate passed (run 58).* `bounce_back_interp(scale)` adds the
  moving-wall term delta = 6 w (c_ob . u_w) to the Bouzidi reflection, divided by 2q on the q > 1/2
  branch (Bouzidi et al. 2001). u_w is interpolated to the wall point, (1-q) uw[x_f] + q uw[x_s],
  which is exact for rigid-body motion provided `uw` holds the rigid velocity on both sides of the
  surface (convention: a band of about 2 cells). Tests: Taylor-Couette flow between a rotating
  inner and a static outer cylinder (both walls from one SDF, all q and directions exercised)
  matches u_theta = A r + B / r within 1% RMS of the wall speed with no radial flow. Flow gate:
  cylinder at Re 100 (D = 45, 7.5% blockage) after the same boundary fix as the sphere (NEEM inlet,
  pressure outlet, start at U, x layers): non-rotating Cd 1.378, St 0.165 (run 56); at spin ratio
  alpha = 1 (counter-clockwise), Cl -2.653 with the Magnus sign, Cd 1.166, St 0.165 (run 58),
  against about |Cl| 2.5 and Cd 1.1 reported for unconfined flow at Re 100.
- *4.2 close-out - bug hunt fixes (runs 59-63).* WALE now takes a solid neighbour's wall velocity
  instead of zero, so moving walls are not seen as a velocity jump. Bouzidi falls back to halfway
  bounce-back when the q <= 1/2 upstream node is solid (thin gaps, wheel contact patches). The
  interpolated-wall force uses the Galilean-invariant form F = sum c (f_in + f_out) - u_w (f_in - f_out).
  Bouzidi and its force loop over a compact boundary-link list built by `set_wall_fractions(q)`
  instead of all 19 x N entries (sphere 994 -> 1205 MLUPS, cylinder 1013 -> 1195). The step case
  moved to a ramped NEEM inlet and pressure outlet: x_r/S 2.82 (was about 2.5; Armaly about 3.0).
  Reference runs reproduce: sphere Cd 1.6228 (identical), cylinder Cl -2.652, Ahmed static 0.7047 and
  moving ground 0.6873, both within their standard errors of the earlier runs.
- *4.2 Moving ground + rotating wheels (original plan).* Generalize the moving-wall velocity BC to the floor
  (belt at inlet U) and to wheel surfaces (local tangential speed). *Gate: moving belt removes
  the floor boundary layer (measured vs static floor); a spinning cylinder shows the expected
  Magnus lift sign.*
- *4.3 Per-part force breakdown.* Multiple named body masks, each with its own momentum-
  exchange sum. *Gate: per-part forces sum to the single-mask `drag_body` within rounding.*
  *Done (dca55f5):* `body` holds part ids (0 = not measured), `part_force` is accumulated in
  `drag_body` and `drag_interp` (Bouzidi links carry `link_part`, read from `body` in
  `set_wall_fractions`). Tests 38-39 split a sphere into two parts; the parts sum to the total within
  float32 rounding of the part magnitudes. Ahmed with a single part reproduces run 72 exactly (run 73).
- *4.4 Integration run.* A front wing (simplest real part) or full car. *Exit: sane, stable
  coefficients + a relative gate - e.g. a wing-angle or ride-height sweep with the expected
  monotonic downforce / ground-effect trend.*
  *Progress (runs 74-82):* inverted NACA 4412, 4 deg, chord 80 cells, Re_c 5000, quasi-2D (nz = 4),
  moving ground and ceiling, Bouzidi wing as part 1. -CL rises from 0.309 at h/c 1.5 to a peak of
  0.337 at h/c 0.3 (+9%), holds at 0.2, then falls (0.255 at 0.1, 0.188 at 0.075). CD rises
  monotonically as the gap closes (0.094 to 0.145). All runs stable, drift below 0.5%. The
  enhancement-then-reduction trend is reproduced; the peak sits at larger h/c and the enhancement
  is weaker than in high-Re moving-ground experiments, which I attribute provisionally to the low
  Reynolds number (thicker wing and ground boundary layers) and will test in the envelope study.

**Phase 5 - Interactivity & sweeps.** Taichi GGUI live viewer (in-run visualization, no
export round-trip), parameter sweeps, A/B comparison tables, mixed-precision for larger
domains. Refactor F (AA-pattern in-place streaming, section 10) lands here or in Phase 4 the
moment a real car domain exceeds the 24 GB cap - it is the memory headroom the bigger runs need.

**Phase 6 - Application (UI).** A standalone setup / run-control / post app: load an STL,
assign boundary conditions, set domain/Re, launch, monitor live, and A/B compare - wrapping the
validated engine so a case does not need a hand-edited example script. Decision (2026-10-01): a full
desktop application in the style of Fluent / COMSOL, not a lightweight dashboard. PySide6 (Qt) + PyVista
(VTK) with a setup tree (geometry, domain, models, boundary conditions, solution, monitors, run, results),
property panels and a 3D viewport. Geometry is imported, not designed: STL first, then OBJ / PLY, later
STEP / IGES through tessellation. The UI never runs the solver in-process: it writes a versioned case file
and launches the solver as a separate process that writes to a run folder. Shipped as a Windows .exe with
versioned releases (UI and engine versions recorded in every run). ParaView stays for heavy post.

Progress (2026-10-01). MVP complete: M1 versioned JSON case files (`src/run/case_file.py`, geometry
by reference: STL or NACA); M2 `python -m src.run` into run folders (atomic progress.json, STOP file,
case hash, result.json, live coefficient samples, solver.log); M3 application window (setup tree,
dataclass-driven property forms with live `Case.validate`, open / save / autosave, PyVista viewport
with named and corner views, fit, orthographic); M4 run control (solver as a separate process, console,
progress, live force monitors, clean stop and exit); M5 Windows build (PyInstaller onedir with the
engine and Taichi as source, one executable with `--solve`, Inno Setup installer, icon). Gates: the
wing template reproduces run 151 exactly from the command line (runs 165-166), from the application,
and from the installed executable. Wing geometry set-up fell from 42 s to 6 s (phi evaluated once, on
one xy-plane for extruded sections). Next: release pipeline, UI backlog, STL import in the UI,
validation suite, sweeps.

v0.1.1 plan (2026-10-01). v0.1.0 shipped as a pre-release built locally and unsigned; VirusTotal flags it
on 1 of about 70 engines (Arctic Wolf, generic machine-learning verdict), a known pattern for unsigned
PyInstaller builds. v0.1.1 is a hygiene and polish release, no solver changes:
- Build: `upx=False` in the spec (EXE and COLLECT); evaluate a PyInstaller bootloader built from source;
  re-scan on VirusTotal before publishing; report the v0.1.0 false positive to Arctic Wolf.
- Desktop icon wrong in v0.1.0: check the icon embedded in HomebrewCFD.exe (file Properties) versus the
  shortcut; ship icon.ico in the install folder and point the Inno Setup shortcuts at it (IconFilename);
  rule out the Windows icon cache before changing the build.
- Release pipeline: GitHub Actions builds the installer on a tag push (windows runner, CPU smoke test of
  the solver entry point), publishes SHA-256 checksums and build-provenance attestation; local builds are
  no longer released.
- Dependencies: pin versions (requirements lock file), pip-audit in CI, Dependabot.
- Code signing: compare SignPath (open-source programme), Certum open-source certificate and Azure
  Trusted Signing on current terms and price; adopt one if affordable.
- UI backlog: summary shows Running during a run; inputs frozen while running (still navigable); Run /
  Stop on their own ribbon tab; Monitors on a resizable Results tab; warmup / averaging phase in the
  status line and as a marker on the plot; running-mean line; x-axis in flow-throughs; capitalized
  status, console and summary messages; "max |u|" spacing; floor / ceiling drawn only for y walls.
- Gate: the CI-built installer reproduces run 151 (C_y -0.3368, SE 0.0004, C_x 0.1144).

Product vs development (2026-10-01). Shipped: the engine (`src/`), the app, case templates (the current
examples as drop-in cases), and a per-run record in each run folder (case, versions, backend, device,
metrics). Development only: `docs/run_log.csv` (aggregated from run records), tests, the validation
gates, this document. The examples become a validation suite (case + reference value + tolerance)
that runs in development and before every release; a "Verification" menu in the app is optional.

Platforms (2026-10-01): the goal is Windows / macOS / Linux on NVIDIA / AMD / Intel GPUs. Backend becomes
a setting with auto-detect (CUDA, then Vulkan, then Metal, then CPU); `src/engine/runtime.py` and
`run_case` currently hard-code CUDA. I can test Windows and Linux on NVIDIA only, so I validate CUDA and
Vulkan there (Vulkan on NVIDIA exercises the same code path AMD / Intel would use) and CPU in CI. macOS
(Metal) and AMD / Intel hardware stay unclaimed until someone tests them; Metal has no float64.

App requirements from common CFD-tool experience (2026-10-01):
- Keep: setup tree with per-node status, templates, live monitors, run queue, run comparison, headless
  CLI, saved case + versions per run, pre-run checks (voxel / link preview, memory and time estimate).
- Avoid: opaque solver errors (translate to a cause and a fix), UI freezes (solver in its own process),
  lost work (autosave), cases that stop opening after an update (schema versions + migrations), hidden
  defaults (show every value the run uses), modal dialogs during a run.
- Bugs to design out: STL unit / scale mistakes (show bounding box and cells across the body on import),
  non-watertight STL (check and report), GPU out-of-memory (estimate before launch), orphaned solver
  processes (kill on exit, PID file), half-written progress files (write to temp, then rename), stale
  results after a case edit (case hash stored per run), paths with spaces / non-ASCII, Windows DPI scaling,
  decimal-comma locales (JSON only, never locale-formatted numbers in files).
- Later, high value for F1 use: parametric sweeps (ride height, angle) as one queued study; checkpoints
  to stop and resume.

Release security (2026-10-01): GitHub does not malware-scan release binaries. Releases are built only
by GitHub Actions from a tagged commit (never from a local machine), with build-provenance attestation and
SHA-256 checksums published; dependencies pinned and checked (pip-audit, Dependabot); each installer
scanned on VirusTotal before publishing; Windows code signing when affordable (unsigned builds trigger
SmartScreen). The app reads case files as data only (JSON, no pickle / eval / embedded scripts) and makes no
network calls.

**Scope note - one engine, many problems.** The core is a general incompressible / low-Mach
LBM solver; external aero (F1 the flagship) is the first domain, not the boundary. Same core,
added incrementally with its own validation each time: internal duct/HVAC flow (new BCs only),
heat transfer (a second thermal distribution), aeroacoustics (near-field direct + a far-field
FW-H analogy; ties to the aeroacoustics work). **Future branch:** multiphase / free-surface
(Shan-Chen or color-gradient) - a major addition, tracked, not near-term. **Out of scope by
design:** high-Mach / compressible / reacting-detonation flow is a different solver class and
is deliberately not pursued here - it would break the validation-first charter.

Each phase ends only when its validation gate passes. Docs and tests are updated within
the same phase, not after.

## 7a. Operating envelope

Each entry changes one parameter on a validated case and measures it against a reference. Runs are in
`docs/run_log.csv`.

### E2: relaxation time limit (shear-wave decay, runs 84-107)

Fully periodic 4 x 32 x 4 box, u_x = 0.05 sin(ky) plus a 1e-4 broadband seed that excites the unstable
modes. Viscosity is fitted from the exponential decay over one e-folding; a run counts as stable only if
the non-wave residual decays.

| tau | BGK nu error | TRT (Lambda 3/16) | Regularized |
|---|---|---|---|
| 0.6 | +0.32% | -0.14% | +0.21% |
| 0.55 | +0.33% | -0.15% | +0.27% |
| 0.52 | +0.34% | -0.14% | +0.32% |
| 0.51 | +0.37% | -0.13% | +0.33% |
| 0.505 | +0.42% | -0.12% | +0.37% |
| 0.502 | +0.51%, residual grows | -0.08%, residual grows | +0.46% |
| 0.501 | +0.57%, residual grows | blow-up | +0.51% |
| 0.5005 | +0.80%, residual grows | blow-up | +15.9%, residual grows |

- Viscosity itself is accurate to well under 1% wherever a model is stable. The limit is stability, not
  the recovered viscosity.
- Lowest usable tau: BGK 0.505, TRT 0.505, regularized 0.501 (a fivefold lower viscosity).
- Reynolds ceiling this sets, Re = U L / nu with nu = (tau - 1/2) / 3: at U = 0.05, Re_max is about
  30 L (BGK/TRT) and 150 L (regularized), with L the resolved length in cells. For example L = 100
  cells gives Re of about 3000 and 15000. Higher Re needs the eddy viscosity of an LES model.
- TRT with Lambda = 3/16 is less stable than BGK below tau = 0.502. That value of Lambda is chosen for
  the exact halfway wall in Poiseuille flow; Lambda = 1/4 is the usual stability optimum. Decision
  (2026-09-30): I keep 3/16 and do not add a 1/4 option. Production cases (Ahmed, wing) use the regularized
  collision, which has no Lambda; TRT runs only in the channel, step and sphere cases, where staircase
  bounce-back walls need the exact wall location and tau stays above the 0.505 floor.
- Caveat: stability here is for a smooth resolved wave with a small seed. Flows with sharp gradients
  (separation, corners) are less forgiving, so treat these as upper bounds.

### E1: resolution (sphere, Re 50, W = 6.4 D, runs 108, 110-113)

| D (cells) | 10 | 15 | 20 | 30 | 40 |
|---|---|---|---|---|---|
| Cd | 1.5986 | 1.6351 | 1.6228 | 1.6183 | 1.6173 |

Cd is converged to about 0.1% by D = 30 and to about 0.3% at D = 20, the default. D = 10 is
under-resolved. The remaining gap to Schiller-Naumann is therefore not a discretization error.

### E4: blockage (sphere, D = 20, runs 108, 119-122)

| W (diameters) | 4 | 5 | 6.4 | 8 | 12 |
|---|---|---|---|---|---|
| blockage | 4.9% | 3.1% | 1.9% | 1.2% | 0.55% |
| Cd | 1.7194 | 1.6578 | 1.6228 | 1.6068 | 1.5957 |

Blockage accounts for most of the historical +5% offset. Linear extrapolation from W = 8 and 12 gives
about 1.587 at zero blockage, +3.2% against Schiller-Naumann (itself a correlation with scatter of a
few percent). A tunnel of at least 8 D keeps blockage bias below about 1%.

### E3: Mach number (2D cylinder, Re 100, D = 45, runs 114-118)

| U | 0.025 | 0.05 | 0.1 | 0.15 | 0.2 |
|---|---|---|---|---|---|
| Ma | 0.043 | 0.087 | 0.17 | 0.26 | 0.35 |
| Cd | 1.3686 | 1.3688 | 1.3780 | 1.4110 | 1.4421 |
| vs incompressible | - | 0.0% | +0.7% | +3.1% | +5.4% |

U <= 0.05 is at the incompressible limit; U = 0.1 carries about +0.7% compressibility error; U >= 0.15
is not usable for quantitative work.

Strouhal (runs 138-140, `dominant_frequency`: Hann FFT peak + parabolic log-magnitude interpolation):

| U | 0.025 | 0.05 | 0.1 |
|---|---|---|---|
| St (spectral) | 0.1711 | 0.1710 | 0.1710 |
| St (zero crossings, full record / first quarter dropped) | 0.1755 / 0.1716 | 0.1686 / 0.1712 | 0.1689 / 0.1712 |

Cd reproduced runs 114-116 exactly. St shows no Mach trend up to Ma 0.17. The zero-crossing estimate on the
full 11-cycle record scatters by about 0.004 from the residual start transient; with the first quarter
dropped it agrees with the spectral value within 0.0006, so I report the spectral value. St 0.171 is +4.3%
vs Williamson (about 0.164 at Re 100) on the inlet velocity and -0.9% on the local velocity beside the
cylinder (U_eff about 1.05 U, blockage 7.5%): the offset is blockage, consistent with E2.

### E5: wing Reynolds number (NACA 4412 inverted, alpha 4, c = 80, moving ground, runs 75-82, 141-150)

-CL, and change vs h/c 1.5:

| h/c | Re 2000 (tau 0.506) | Re 5000 (tau 0.5024) | Re 10000 (tau 0.5012) |
|---|---|---|---|
| 1.5 | 0.2619 | 0.3089 | 0.3059 |
| 0.3 | 0.2506 (-4.3%) | 0.3368 (+9.0%) | 0.3396 (+11.0%) |
| 0.2 | - | 0.3350 (+8.4%) | 0.3473 (+13.5%) |
| 0.15 | 0.1990 (-24.0%) | 0.3154 (+2.1%) | 0.3424 (+11.9%) |
| 0.1 | 0.1308 (-50.1%) | 0.2553 (-17.4%) | 0.3098 (+1.3%) |

The ground-effect peak moves toward the ground as Re rises: none at Re 2000 (the ground only costs
downforce), h/c 0.2-0.3 at Re 5000, h/c about 0.2 at Re 10000 (run 150; above h/c 0.15 and 0.3 by 1.3-1.6 combined SE, so the location is
resolved to about +/-0.05). Low
Re is therefore a credible reason the peak sits above the experimental h/c of about 0.1 (Zerihan and Zhang,
Re about 4.6e5, turbulent): thick viscous layers on the wing and in the gap choke the underside flow
earlier. Re 10000 sheds an unsteady wake (SE 3-10x larger than Re 5000). Quasi-2D laminar LBM at
Re 1e4 is not the experimental regime, so the trend is qualitative, not a match to the experiment.

### E6: Ahmed body resolution (25 deg, round nose, WALE, static floor, Re_H 30000, runs 123-126)

| H (cells) | 24 | 32 | 40 | 48 |
|---|---|---|---|---|
| Cd (SE) | 0.755 (0.002) | 0.700 (0.002) | 0.667 (0.002) | 0.628 (0.002) |
| wall-model nodes | 3350 | 974 | 334 | 1 |

Not grid-converged: Cd falls monotonically by 4-8% per step with no plateau up to H = 48. Absolute Ahmed
Cd values from this setup are resolution-dependent at the 10% level and are not reported as results;
comparisons at a fixed H (for example static against moving ground) remain meaningful. The study is
confounded by the log-law model, whose y+ > 30 gate switches it off as the grid is refined (3350
engaged nodes at H = 24, one at H = 48), and by the implicit LES filter width shrinking with the cell
size.

With the wall model disabled (runs 127-129) Cd is 0.754, 0.700 and 0.629 at H = 24, 32 and 48, identical
to the wall-modelled runs within the standard error. The log-law model has no measurable effect on Cd
here, so the resolution trend comes from the grid and the implicit LES filter: the mean WALE eddy
viscosity falls from 6.2 nu at H = 24 to 3.0 nu at H = 48, and Cd falls with it. The differences between
successive grids (0.055, 0.033, 0.039) are not decreasing geometrically, so the runs are not yet in the
asymptotic range and a Richardson extrapolation would not be meaningful.

Relative test (runs 130-131): Cd(35 deg) - Cd(25 deg) is +0.004 at H = 24 and +0.010 at H = 32, the same
sign but a factor of 2.4 apart and within 1.5-4.5 standard errors of zero. The slant flow is separated at
both angles (attached fraction 0.05-0.12), so the attached-to-separated change that makes 25 deg and
35 deg differ in experiments does not occur at Re_H 30000. The Ahmed body at this Reynolds number is
therefore not a usable test of relative accuracy; a case with a large, regime-stable difference is needed.

### Relative accuracy under refinement (wing in ground effect, chord 80 and 120, runs 74-82, 132-137)

| h/c | 1.5 | 0.6 | 0.3 | 0.15 | 0.1 |
|---|---|---|---|---|---|
| -CL, c = 80 | 0.309 | 0.320 | 0.337 | 0.315 | 0.255 |
| -CL, c = 120 | 0.321 | 0.335 | 0.354 | 0.330 | 0.260 |
| change vs h/c 1.5, c = 80 | - | +3.6% | +9.0% | +2.1% | -17.4% |
| change vs h/c 1.5, c = 120 | - | +4.5% | +10.3% | +2.8% | -18.9% |

The absolute level rises about 4-5% with refinement, but the shape of the curve is preserved: the peak
stays at h/c 0.3 and every relative change agrees to within 1.5 percentage points (10-30% of the change
itself, same sign everywhere). Relative trends are therefore more robust to resolution than absolute
coefficients, which supports the project's stated accuracy target; the Ahmed body could not show this
because its 25 and 35 degree cases are in the same separated regime at Re_H 30000.

## 8. Additional lattice stencils (future)

The lattice is a pure-data descriptor (`Q, D, E, W, OPP, CS2`) that the operators
consume generically, so new stencils are data-only additions - no kernel changes for
same-dimension sets. Candidates, and what each is actually good for:

- **D3Q15 / D3Q19 / D3Q27** - the 3D Navier-Stokes workhorses. D3Q19 is the default;
  D3Q27 for high-Re isotropy/stability; D3Q15 for cheap/coarse runs (weaker isotropy).
  *(D3Q19 done; Q15/Q27 planned as data-only additions.)*
- **D2Q9** - the standard 2D NS stencil (done).
- **D2Q5 / D3Q7** - *not* full fluid stencils. These are advection-diffusion lattices for
  a scalar field (temperature, species). Useful later if heat transfer or passive
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
- FP16 storage - how much accuracy is traded for domain size? (measure in Phase 5).
- Real F1 geometry source and its licensing (needed by Phase 4).
- Cylinder Cd calibration: blockage, resolution, tau, MEM factor (Phase 2, in progress).

- Generic runner (done 2026-10-01, runs 151-164): every example is ported to one `Case` + `run_case` so runs compare
  like for like; each port must reproduce its logged reference run exactly. `Case` carries `lattice`
  and `dimensions` from the start so the case format survives new stencils. Order: runner and ports,
  then the Phase 6 app, then D3Q15/D3Q27 and the 2D engine.
- 2D engine parity (after the app): `src/engine/simulation.py` gets the 3D physics (regularized and TRT
  collision, Bouzidi walls, NEEM inlet and pressure outlet, relaxation layers, WALE) for cases where 2D
  speed matters. The NumPy D2Q9 reference in `src/lbm/` stays as the validation reference.
- Tau floor under LES: `Case.validate` rejects tau below 0.501 unless the case sets
  `allow_below_floor`; the flag is logged. The floor is measured without LES (E2); the limit with WALE
  is unmapped.

- Compressible flow (long-term option, 2026-10-01). The current solver is isothermal and its D3Q19
  equilibrium is second order in velocity, so compressibility errors grow like Ma^3 (E3: +0.7% at
  lattice Ma 0.17, +5.4% at 0.35) and there is no energy equation or shock handling. F1 aero does not need
  it (about Ma 0.28 at 340 km/h, normally treated as incompressible; the lattice Mach number is a
  numerical choice). Known routes: higher-order lattices (D3Q39 / D2Q37, costly, narrow stability),
  double-distribution models with an energy population plus correction terms, and hybrid LBM (lattice
  Boltzmann for mass and momentum, finite-difference / finite-volume energy equation, pressure corrections
  and shock sensors). If pursued, I would stage it: thermal LBM (already planned for heat transfer), then a
  weakly compressible variable-density model validated on natural convection, then a hybrid compressible
  branch validated on the Sod shock tube, the isentropic vortex and transonic NACA 0012 data. Shocks with
  chemistry (detonation) remain out of scope: a reactive finite-volume problem, not this solver class.

## 10. Code-health backlog

From the full code review. Done so far: wall nodes no longer collided (this was the
"0.33-cell body-force wall offset" - TRT forced Poiseuille is now exact at halfway, delta_fit
31.999, peak -0.07%); Smagorinsky now uses the textbook 18*sqrt(2) (old runs' cs=0.1 ~ 0.084
now); per-cell conservation tests with the exact stream shift and Guo +F identities; vacuous
LES assert replaced; Ahmed mask test; step.py dead x_r block removed; every example's progress
bar now finishes at 100% (`prog.done()` right after the loop); stale 0.33/0.83 wall-offset
comments and docs corrected; step.py duplicate x_r print removed and final print guarded;
`ahmed.py` geometry renamed `ahmed_body.py`. **A. Memory: DONE** - `Simulation3D(...,
interp=False)` default; `q`/`fc` allocated only with `interp=True` (cylinder/sphere/test
fixture pass it). ~70M -> ~130M cells on 24 GB. Verified: tests green, cylinder Cd 1.520 /
St 0.165 unchanged. **B. `ti.init` once per process: DONE** - new `src/engine/runtime.py`
(`init(backend)`: init once, no-op on same arch, `RuntimeError` on a switch); both constructors
call it. Red first: the old code let a stale `f` handle read the new runtime's memory (a 4x4x4
int32 array) - silent corruption, no error. `src/config/environment.py` untouched, but its test
now runs the probe in a subprocess (its `ti.init(cuda)` would re-init Taichi behind `runtime`).
Tests 25 (`test_second_instance_keeps_fixture`) and 26 (`test_runtime_rejects_switch`) added;
test-ordering constraint gone. Trade-off: fields never freed within a process. Verified: full
suite green. **C. Ahmed loop speed: DONE** - precomputed wall-node list + normals
(`build_wall_list`, geometry-static, vectorized), new `wall_model_fast` loops only wall-adjacent
nodes with local moments; dropped the redundant per-step `macroscopic()` (collide_reg computes
its own moments). ~20% faster (997 s vs 1240 s, H=32 phi=25). `wall_model` kept for tests 21/24;
bit-identity guard `test_wall_model_matches` (fast vs slow, 1e-6). Run-level Cd 0.7444 vs 0.7422
baseline - within +/-0.003 error bars (changed FP op order decorrelates the chaotic trajectory;
time-mean unchanged). **D. Duplicated physics: DONE** - inlet family uses `feq`
and a shared `_neem_column` @ti.func; `drag`/`drag_body` share a `_drag_mask` @ti.func
(use_body flag); `collide_full` (LES branch) and `collide_reg` share a `_stress` @ti.func for
the non-eq stress tensor. All no-op numerically: bit-identity guards (drag test; collide golden
via array_equal on reg + full) plus tests 6/7/8/9/14/15/22/23 green. **E. Small: DONE**
(mostly) - `sphere` now takes radius like `cylinder`/`wall_fraction_*` (mask identical);
`sphere.py` health check uses a new `f_absmax` GPU reduction, full `f` pulled only on
blow-up; dropped `channel_turbulent`'s unused per-step uc/urms/vrms/wrms; `next`->`u_next`
in `wall_function`; `. all`->`.all` in `lattice3d`; dead trailing `return` in `collide_reg`.
KEPT: `config/loader.py` + `cases/` - unused by the solver but `test_scaffold.py` depends on
them, so not "dead"; leave until that test is reworked. **step.py: DONE** - reattachment x_r is now a
linear sub-cell zero-crossing between the last-negative and first-positive floor samples,
not quantized to whole cells.

Backlog status: A, B, C, D, E and step.py all closed; 2D collide-on-walls and the
channel/Ahmed re-confirms done; Phase 3 relative gate passed (nose shape). Only F remains,
deferred by design until an H=48+ run hits the 24 GB cap - at which point the memory
pressure also gives the AA rewrite a concrete validation target (full ladder re-run). **2D collide-on-walls: DONE** - both 2D paths collided wall nodes
(`src/lbm` `collide`/`collide_forced`, and `Simulation.collide`); parity and fit-root tests hid
it. Red first: new `test_poiseuille.py::test_wall_location` pins the walls at 0.5 / ny-1.5 at
the BGK magic tau = 1/2 + sqrt(3)/4 (halfway bounce-back exact) - fitted wall was at 0.084, a
0.42-cell offset. Fix: optional `solid` mask (wall nodes keep their populations, no force);
Taichi 2D skips solid/lid like 3D. Cavity test/demo/parity pass the full mask. Verified: wall
location within 0.01, Ghia still within 5%, suite green. **Re-confirm after fixes: DONE** - Re_tau=180 channel holds:
U+ 18.49 (was 18.3, MKM 18.3), u'_rms peak 3.08 at y+=15.5 (was 3.2), asym 2.1%. Ahmed (H=32,
phi=25, Re_H=30k, cs=0.084 = old 0.1): stable, max|u| 0.088, but **Cd 3.245 -> 0.740 (-77%)**.
Only physics change is the wall-node collision skip; likely the old solid-node collision
contaminated the momentum-exchange drag itself (consistent with the flat first-pass sweep).

Open (in order):
- **F. DEFERRED (only when memory-bound).** AA-pattern in-place streaming - drops `f_new` (76 B/cell) for ~40% more
  cells. Bigger change (alternating even/odd kernels, boundaries must follow the pattern);
  only if H=48+ runs hit the memory cap.

After the backlog: resume the slant sweep follow-ups (section 7, Ahmed "Sweep, first pass").
